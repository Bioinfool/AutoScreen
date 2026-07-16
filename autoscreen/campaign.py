"""Asynchronous, plate-based active-learning campaign with resumable state.

This is the async replacement for MolPAL's synchronous
`scores = objective(mols); model.update(...)` loop. Each round:

    1. train per-objective surrogates on the accumulated QC-passed labels
    2. score the unlabeled pool and build a 96-well PlateLayout (plate.py)
    3. submit the plate to the RobotClient
    4. poll until the job is done, collecting partial results as they arrive
    5. accept only COMPLETED + qc_passed experimental/replicate wells into training
    6. checkpoint the campaign state to disk

The checkpoint (JSON) records the labeled set, per-round history and the RNG
position, so a campaign interrupted mid-run can be resumed from the last
completed round. Controls (positive/negative/blank) are measured for QC/metrics
but never added to the surrogate training set.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .acquire import _normalize
from .data import MOODataset
from .experiment import WellState
from .metrics import hypervolume, make_ref_point, pareto_mask
from .model import MultiOutputRFSurrogate
from .plate import PlateConfig, select_plate
from .robot_client import RobotClient


@dataclass
class RoundRecord:
    round: int
    n_labeled: int
    hv_frac: float
    pareto_recall: float
    submitted: int
    completed: int
    failed: int
    qc_rejected: int
    replicate_rmse: float | None
    polls: int


@dataclass
class CampaignState:
    campaign_id: str
    seed: int
    round: int
    labeled_idx: list[int]
    positive_idx: list[int]
    negative_idx: list[int]
    history: list[dict] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CampaignState":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**d)


class Campaign:
    def __init__(
        self,
        ds: MOODataset,
        client: RobotClient,
        plate_config: PlateConfig,
        campaign_id: str = "screen_001",
        seed: int = 0,
        checkpoint: str | Path = "runs/campaign_state.json",
        max_polls: int = 50,
        verbose: bool = True,
    ):
        self.ds = ds
        self.client = client
        self.plate_config = plate_config
        self.checkpoint = Path(checkpoint)
        self.max_polls = max_polls
        self.verbose = verbose

        self.ref_point = make_ref_point(ds.Y)
        gmask = pareto_mask(ds.Y)
        self.global_front_idx = set(np.where(gmask)[0].tolist())
        self.global_hv = hypervolume(ds.Y[gmask], self.ref_point)

        # stock availability: a fixed ~5% of the library is "out of stock"
        stock_rng = np.random.default_rng(1234)
        self.stock_available = stock_rng.random(ds.n) >= 0.05

        if self.checkpoint.exists():
            self.state = CampaignState.load(self.checkpoint)
            if self.verbose:
                print(f"[resume] loaded round {self.state.round}, "
                      f"{len(self.state.labeled_idx)} labeled from {self.checkpoint}")
        else:
            self.state = self._init_state(campaign_id, seed)

    def _init_state(self, campaign_id: str, seed: int) -> CampaignState:
        rng = np.random.default_rng(seed)
        # controls: fix reference actives (best activity) and inactives (worst activity)
        order = np.argsort(-self.ds.Y[:, 0])
        positive = order[: self.plate_config.n_positive].tolist()
        negative = order[-self.plate_config.n_negative:].tolist()
        # seed the labeled set with a small random init batch (excluding controls)
        control_set = set(positive) | set(negative)
        candidates = [i for i in range(self.ds.n) if i not in control_set]
        init_k = max(1, int(0.01 * self.ds.n))
        init = rng.choice(candidates, size=init_k, replace=False).tolist()
        return CampaignState(
            campaign_id=campaign_id,
            seed=seed,
            round=0,
            labeled_idx=init,
            positive_idx=positive,
            negative_idx=negative,
        )

    def _labeled_mask(self) -> np.ndarray:
        m = np.zeros(self.ds.n, dtype=bool)
        m[self.state.labeled_idx] = True
        return m

    def _metrics(self) -> tuple[float, float]:
        labeled = self._labeled_mask()
        hv = hypervolume(self.ds.Y[labeled], self.ref_point)
        found = len(self.global_front_idx & set(self.state.labeled_idx))
        return (
            hv / self.global_hv if self.global_hv > 0 else 0.0,
            found / max(1, len(self.global_front_idx)),
        )

    def run_round(self) -> RoundRecord:
        rnd = self.state.round + 1
        rng = np.random.default_rng(self.state.seed * 1000 + rnd)
        labeled = self._labeled_mask()
        pool_idx = np.where(~labeled)[0]

        # 1. train surrogates on QC-passed labels only
        model = MultiOutputRFSurrogate(self.ds.n_objectives, seed=self.state.seed)
        model.fit(self.ds.X[labeled], self.ds.Y[labeled])
        means, stds = model.predict(self.ds.X[pool_idx])

        # 2. scalarize (equal weights, optimistic) and build a plate
        norm_range = (self.ds.Y[labeled].min(axis=0), self.ds.Y[labeled].max(axis=0))
        beta = 0.5
        opt = _normalize(means + beta * stds, norm_range)
        scores = opt.mean(axis=1)
        pred_sa_ease = means[:, 2]

        layout = select_plate(
            round=rnd,
            pool_idx=pool_idx,
            scores=scores,
            pred_sa_ease=pred_sa_ease,
            fps_pool=self.ds.X[pool_idx],
            smis_pool=[self.ds.smis[i] for i in pool_idx],
            positive_idx=self.state.positive_idx,
            negative_idx=self.state.negative_idx,
            all_smis=self.ds.smis,
            config=self.plate_config,
            stock_available=self.stock_available[pool_idx],
            rng=rng,
        )

        # 3. submit
        job_id = self.client.submit_plate(self.state.campaign_id, layout)

        # 4. poll until done (or max_polls), collecting results
        polls = 0
        status = self.client.fetch_results(job_id)
        while not status.done and polls < self.max_polls:
            polls += 1
            status = self.client.fetch_results(job_id)

        # 5. QC gate: accept COMPLETED + qc_passed experimental wells only
        completed = failed = qc_rejected = 0
        replicate_errs: list[float] = []
        primary_activity: dict[str, float] = {}  # compound_id -> measured activity
        new_labels: dict[int, list[float]] = {}
        control_set = self.global_control_set()

        for r in status.results:
            if r.state is WellState.COMPLETED and r.qc_passed and r.kind != "blank":
                completed += 1
            elif r.state is WellState.FAILED:
                failed += 1
            elif r.state is WellState.QC_REJECTED:
                qc_rejected += 1
            if r.usable and r.kind == "experimental":
                primary_activity[r.compound_id] = r.values[0]

        for r in status.results:
            if not r.usable or r.pool_idx < 0:
                continue
            if r.kind == "replicate":
                # QC reproducibility check vs the primary reading; do not re-add
                if r.compound_id in primary_activity:
                    replicate_errs.append(abs(r.values[0] - primary_activity[r.compound_id]))
                continue
            if r.kind != "experimental" or r.pool_idx in control_set:
                continue  # controls measured for QC but never trained on
            new_labels[r.pool_idx] = r.values

        # 6. commit new labels (controls excluded)
        for gidx in new_labels:
            if gidx not in self.state.labeled_idx:
                self.state.labeled_idx.append(gidx)

        self.state.round = rnd
        hv_frac, recall = self._metrics()
        rep_rmse = (
            float(np.sqrt(np.mean(np.square(replicate_errs)))) if replicate_errs else None
        )
        rec = RoundRecord(
            round=rnd,
            n_labeled=len(self.state.labeled_idx),
            hv_frac=hv_frac,
            pareto_recall=recall,
            submitted=len(layout.wells),
            completed=completed,
            failed=failed,
            qc_rejected=qc_rejected,
            replicate_rmse=rep_rmse,
            polls=polls,
        )
        self.state.history.append(asdict(rec))
        self.state.save(self.checkpoint)

        if self.verbose:
            rr = f"{rep_rmse:.3f}" if rep_rmse is not None else "n/a"
            print(
                f"[{self.state.campaign_id}] r{rnd:02d} "
                f"n={rec.n_labeled:5d} hv_frac={hv_frac:.3f} pareto={recall:.3f} "
                f"| plate: ok={completed} fail={failed} qc_rej={qc_rejected} "
                f"polls={polls} rep_rmse={rr}"
            )
        return rec

    def global_control_set(self) -> set[int]:
        return set(self.state.positive_idx) | set(self.state.negative_idx)

    def run(self, n_rounds: int) -> list[dict]:
        target = self.state.round + n_rounds
        while self.state.round < target:
            pool_left = self.ds.n - len(self.state.labeled_idx)
            if pool_left <= 0:
                break
            self.run_round()
        return self.state.history
