"""CampaignManager: active-learning orchestration over a pluggable Executor."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from autoscreen.core.acquisition import AcquisitionStrategy, build_acquisition
from autoscreen.core.constraints import ConstraintManager, PlateConfig
from autoscreen.core.library import CandidateLibrary
from autoscreen.core.metrics import hypervolume, make_ref_point, pareto_mask
from autoscreen.core.model import MultiOutputRFSurrogate, SurrogateModel
from autoscreen.core.observations import ObservationStore
from autoscreen.core.types import ItemKind, Job, JobItem, Observation, WellState
from autoscreen.executors.base import Executor
from autoscreen.logging_utils import get_logger

log = get_logger("campaign")


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
    job_id: str
    acquisition: str


@dataclass
class CampaignState:
    campaign_id: str
    seed: int
    round: int
    history: list[dict] = field(default_factory=list)
    positive_idx: list[int] = field(default_factory=list)
    negative_idx: list[int] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CampaignState":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class CampaignManager:
    def __init__(
        self,
        library: CandidateLibrary,
        executor: Executor,
        acquisition: AcquisitionStrategy | str = "greedy",
        *,
        campaign_id: str = "campaign",
        seed: int = 0,
        batch_size: int = 100,
        init_frac: float = 0.01,
        beta: float = 1.0,
        checkpoint_dir: str | Path = "runs/campaign",
        n_estimators: int = 100,
        plate: PlateConfig | None = None,
        constraints: ConstraintManager | None = None,
        use_plate_layout: bool = False,
        max_polls: int = 200,
        resume: bool = False,
    ):
        if library.Y_hidden is None and executor.kind == "replay":
            raise ValueError("Replay campaigns require hidden labels on the library")
        self.library = library
        self.executor = executor
        self.batch_size = batch_size
        self.init_frac = init_frac
        self.max_polls = max_polls
        self.use_plate_layout = use_plate_layout
        self.plate = plate or PlateConfig()
        self.constraints = constraints or ConstraintManager(self.plate)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.store_path = self.checkpoint_dir / "observations.json"
        self.state_path = self.checkpoint_dir / "campaign_state.json"

        if isinstance(acquisition, str):
            self.acquisition = build_acquisition(acquisition, beta=beta)
        else:
            self.acquisition = acquisition

        self.n_obj = library.n_objectives if library.Y_hidden is not None else 3
        self.model: SurrogateModel = MultiOutputRFSurrogate(
            self.n_obj, n_estimators=n_estimators, seed=seed
        )

        if self.use_plate_layout:
            self.plate.validate()

        if library.Y_hidden is not None:
            self.ref_point = make_ref_point(library.Y_hidden)
            gmask = pareto_mask(library.Y_hidden)
            self.global_front_idx = set(np.where(gmask)[0].tolist())
            self.global_hv = hypervolume(library.Y_hidden[gmask], self.ref_point)
        else:
            self.ref_point = np.zeros(self.n_obj)
            self.global_front_idx = set()
            self.global_hv = 1.0

        if resume:
            if not (self.state_path.exists() and self.store_path.exists()):
                raise FileNotFoundError(
                    f"Cannot resume: missing checkpoint under {self.checkpoint_dir}"
                )
            self.state = CampaignState.load(self.state_path)
            self.store = ObservationStore.load(self.store_path)
            log.info(
                "Resumed campaign %s at round %s (%s labeled)",
                self.state.campaign_id,
                self.state.round,
                len(self.store),
            )
        else:
            if self.state_path.exists() or self.store_path.exists():
                raise FileExistsError(
                    f"Checkpoint already exists at {self.checkpoint_dir}; "
                    "pass resume=True / --resume, or choose a new checkpoint_dir"
                )
            self.store = ObservationStore()
            self.state = self._bootstrap_state(campaign_id, seed)
            self._checkpoint()

    def _bootstrap_state(self, campaign_id: str, seed: int) -> CampaignState:
        rng = np.random.default_rng(seed)
        order = (
            np.argsort(-self.library.Y_hidden[:, 0])
            if self.library.Y_hidden is not None
            else np.arange(self.library.n)
        )
        positive = order[: self.plate.n_positive].tolist()
        negative = order[-self.plate.n_negative:].tolist()
        control = set(positive) | set(negative)
        candidates = [i for i in range(self.library.n) if i not in control]
        init_k = max(1, int(self.init_frac * self.library.n))
        init_idxs = rng.choice(candidates, size=min(init_k, len(candidates)), replace=False).tolist()

        # Temporary state needed for plate layout helpers during bootstrap
        self.state = CampaignState(
            campaign_id=campaign_id,
            seed=seed,
            round=0,
            positive_idx=positive,
            negative_idx=negative,
            history=[],
        )
        job = self._make_job(0, init_idxs, campaign_id, kind_all=ItemKind.EXPERIMENTAL)
        jid = self.executor.submit(job)
        status = self.executor.wait(jid, max_polls=self.max_polls)
        self.store.add_many(status.observations)
        completed = failed = qc_rej = 0
        for o in status.observations:
            if o.state is WellState.COMPLETED and o.qc_passed and o.kind is not ItemKind.BLANK:
                completed += 1
            elif o.state is WellState.FAILED:
                failed += 1
            elif o.state is WellState.QC_REJECTED:
                qc_rej += 1
        hv_frac, recall = self._metrics()
        self.state.history.append(
            {
                "round": 0,
                "n_labeled": len(self.store),
                "hv_frac": hv_frac,
                "pareto_recall": recall,
                "submitted": len(job.items),
                "completed": completed,
                "failed": failed,
                "qc_rejected": qc_rej,
                "job_id": jid,
                "acquisition": "init",
            }
        )
        return self.state

    def _checkpoint(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.store.save(self.store_path)
        if hasattr(self, "state"):
            self.state.save(self.state_path)

    def _metrics(self) -> tuple[float, float]:
        if self.library.Y_hidden is None or len(self.store) == 0:
            return 0.0, 0.0
        idxs = self.store.labeled_indices
        Y = self.library.Y_hidden[idxs]
        hv = hypervolume(Y, self.ref_point)
        found = len(self.global_front_idx & set(idxs))
        return (
            hv / self.global_hv if self.global_hv > 0 else 0.0,
            found / max(1, len(self.global_front_idx)),
        )

    def _make_job(
        self,
        rnd: int,
        pool_indices: list[int],
        campaign_id: str,
        kind_all: ItemKind = ItemKind.EXPERIMENTAL,
    ) -> Job:
        items: list[JobItem] = []
        if self.use_plate_layout:
            items = self._plate_items(rnd, pool_indices)
        else:
            for i, gidx in enumerate(pool_indices):
                items.append(
                    JobItem(
                        item_id=f"r{rnd}-i{i}",
                        smiles=self.library.smis[gidx],
                        pool_idx=gidx,
                        kind=kind_all,
                        compound_id=f"CMP{gidx:07d}",
                    )
                )
        job_id = f"{campaign_id}-r{rnd}-{uuid.uuid4().hex[:8]}"
        return Job(
            job_id=job_id,
            campaign_id=campaign_id,
            round=rnd,
            items=items,
            executor_kind=self.executor.kind,
            idempotency_key=job_id,
        )

    def _control_indices(self) -> set[int]:
        return set(self.state.positive_idx) | set(self.state.negative_idx)

    def _unavailable_mask(self) -> np.ndarray:
        """Labeled experimental + permanently reserved control wells."""
        labeled = self.store.mask(self.library.n)
        for i in self._control_indices():
            if 0 <= i < self.library.n:
                labeled[i] = True
        return labeled

    def _plate_items(self, rnd: int, exp_indices: list[int]) -> list[JobItem]:
        self.plate.validate()
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if self.plate.rows > len(alphabet):
            raise ValueError(f"Plate rows={self.plate.rows} exceeds letter labels")
        letters = alphabet[: self.plate.rows]
        items: list[JobItem] = []
        w = 0

        def wid() -> str:
            nonlocal w
            if w >= self.plate.capacity:
                raise ValueError(
                    f"Well index {w} exceeds plate capacity {self.plate.capacity}"
                )
            s = f"{letters[w // self.plate.cols]}{w % self.plate.cols + 1:02d}"
            w += 1
            return s

        if len(exp_indices) > self.plate.n_experimental:
            log.warning(
                "Truncating experimental batch %d -> %d to fit plate layout",
                len(exp_indices),
                self.plate.n_experimental,
            )
        exp_wells: list[str] = []
        for gidx in exp_indices[: self.plate.n_experimental]:
            well = wid()
            exp_wells.append(well)
            items.append(
                JobItem(
                    item_id=f"r{rnd}-{well}",
                    well_id=well,
                    smiles=self.library.smis[gidx],
                    pool_idx=gidx,
                    kind=ItemKind.EXPERIMENTAL,
                    compound_id=f"CMP{gidx:07d}",
                )
            )
        for gidx in self.state.positive_idx[: self.plate.n_positive]:
            well = wid()
            items.append(
                JobItem(
                    item_id=f"r{rnd}-{well}", well_id=well,
                    smiles=self.library.smis[gidx], pool_idx=gidx,
                    kind=ItemKind.POSITIVE, compound_id=f"CMP{gidx:07d}",
                )
            )
        for gidx in self.state.negative_idx[: self.plate.n_negative]:
            well = wid()
            items.append(
                JobItem(
                    item_id=f"r{rnd}-{well}", well_id=well,
                    smiles=self.library.smis[gidx], pool_idx=gidx,
                    kind=ItemKind.NEGATIVE, compound_id=f"CMP{gidx:07d}",
                )
            )
        for _ in range(self.plate.n_blank):
            well = wid()
            items.append(
                JobItem(item_id=f"r{rnd}-{well}", well_id=well, smiles="", pool_idx=-1, kind=ItemKind.BLANK)
            )
        rng = np.random.default_rng(self.state.seed * 1000 + rnd)
        if exp_indices and self.plate.n_replicate > 0:
            n_rep = min(self.plate.n_replicate, len(exp_wells))
            picks = rng.choice(len(exp_wells), size=n_rep, replace=False)
            for j in picks:
                gidx = exp_indices[j]
                well = wid()
                items.append(
                    JobItem(
                        item_id=f"r{rnd}-{well}", well_id=well,
                        smiles=self.library.smis[gidx], pool_idx=gidx,
                        kind=ItemKind.REPLICATE, compound_id=f"CMP{gidx:07d}",
                        replicate_of=exp_wells[j],
                    )
                )
        return items

    def run_round(self) -> RoundRecord:
        rnd = self.state.round + 1
        rng = np.random.default_rng(self.state.seed * 1000 + rnd)
        labeled = self._unavailable_mask()
        pool_idx = np.where(~labeled)[0]
        if len(pool_idx) == 0:
            raise RuntimeError("No unlabeled compounds left")

        X, Y = self.store.matrix(self.library.X, self.n_obj)
        self.model.fit(X, Y)
        means, stds = self.model.predict(self.library.X[pool_idx])

        # feasibility filter
        feas = self.constraints.feasible_mask(
            len(pool_idx),
            pred_sa_ease=means[:, 2] if means.shape[1] > 2 else None,
            pool_global_idx=pool_idx,
        )
        feas_local = np.where(feas)[0]
        if len(feas_local) == 0:
            feas_local = np.arange(len(pool_idx))

        k = self.plate.n_experimental if self.use_plate_layout else self.batch_size
        sel = self.acquisition.select(
            pool_idx[feas_local],
            means[feas_local],
            stds[feas_local],
            k,
            labeled_Y=Y,
            ref_point=self.ref_point,
            rng=rng,
        )
        # optional diversity re-rank using constraint manager on selected scores
        if self.use_plate_layout and sel.scores:
            # map back to local feas indices for fps
            score_map = {g: s for g, s in zip(sel.pool_indices, sel.scores)}
            # diversify among acquisition candidates
            cand = np.asarray(sel.pool_indices)
            local = np.array([int(np.where(pool_idx == g)[0][0]) for g in cand])
            scores = np.array([score_map[g] for g in cand])
            div_local = self.constraints.diversify(
                scores, self.library.X[cand], min(k, len(cand))
            )
            chosen = [int(cand[i]) for i in div_local]
        else:
            chosen = list(sel.pool_indices)

        job = self._make_job(rnd, chosen, self.state.campaign_id)
        jid = self.executor.submit(job)
        status = self.executor.wait(jid, max_polls=self.max_polls)

        completed = failed = qc_rej = 0
        for o in status.observations:
            if o.state is WellState.COMPLETED and o.qc_passed and o.kind is not ItemKind.BLANK:
                completed += 1
            elif o.state is WellState.FAILED:
                failed += 1
            elif o.state is WellState.QC_REJECTED:
                qc_rej += 1
        n_new = self.store.add_many(status.observations)

        self.state.round = rnd
        hv_frac, recall = self._metrics()
        rec = RoundRecord(
            round=rnd,
            n_labeled=len(self.store),
            hv_frac=hv_frac,
            pareto_recall=recall,
            submitted=len(job.items),
            completed=completed,
            failed=failed,
            qc_rejected=qc_rej,
            job_id=jid,
            acquisition=self.acquisition.name,
        )
        self.state.history.append(asdict(rec))
        self._checkpoint()
        log.info(
            "[%s] r%02d n=%d hv_frac=%.3f pareto=%.3f new=%d ok=%d fail=%d qc=%d",
            self.state.campaign_id, rnd, rec.n_labeled, hv_frac, recall, n_new,
            completed, failed, qc_rej,
        )
        return rec

    def run(self, n_rounds: int) -> list[dict]:
        target = self.state.round + n_rounds
        while self.state.round < target:
            if self.library.n - len(self.store) <= 0:
                break
            self.run_round()
        return self.state.history
