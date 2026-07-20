"""CampaignManager: async-capable active-learning orchestration."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from autoscreen.core.acquisition import AcquisitionStrategy, build_acquisition
from autoscreen.core.benchmark import BenchmarkEvaluator
from autoscreen.core.candidate_state import CandidateStateStore
from autoscreen.core.constraints import ConstraintManager, PlateConfig
from autoscreen.core.jobs import JobLifecycle, JobRecord, JobStore
from autoscreen.core.library import CandidateLibrary
from autoscreen.core.metrics import make_ref_point
from autoscreen.core.model import MultiOutputRFSurrogate, SurrogateModel
from autoscreen.core.objectives import ObjectiveSchema, default_schema
from autoscreen.core.observations import ObservationStore
from autoscreen.core.types import ItemKind, Job, JobItem, Observation, WellState
from autoscreen.executors.base import Executor
from autoscreen.logging_utils import get_logger

log = get_logger("campaign")


@dataclass
class CampaignState:
    campaign_id: str
    seed: int
    round: int  # completed acquisition rounds (0 = init only / in progress)
    history: list[dict] = field(default_factory=list)
    positive_idx: list[int] = field(default_factory=list)
    negative_idx: list[int] = field(default_factory=list)
    init_done: bool = False
    schema: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CampaignState":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class CampaignManager:
    """Decision layer: never holds hidden labels; only observations + static props."""

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
        schema: ObjectiveSchema | None = None,
        evaluator: BenchmarkEvaluator | None = None,
        max_active_jobs: int = 2,
        positive_idx: list[int] | None = None,
        negative_idx: list[int] | None = None,
    ):
        # Scientific boundary: library must not expose Y_hidden
        if hasattr(library, "Y_hidden") and getattr(library, "Y_hidden") is not None:
            raise ValueError(
                "CandidateLibrary must not carry Y_hidden; use ReplayExecutor(oracle=...) "
                "and BenchmarkEvaluator instead"
            )

        self.library = library
        self.executor = executor
        self.schema = schema or library.schema or default_schema()
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.init_frac = init_frac
        self.max_polls = max_polls
        self.max_active_jobs = max(1, int(max_active_jobs))
        self.use_plate_layout = use_plate_layout
        self.plate = plate or PlateConfig()
        self.constraints = constraints or ConstraintManager(self.plate)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.store_path = self.checkpoint_dir / "observations.json"
        self.state_path = self.checkpoint_dir / "campaign_state.json"
        self.jobs_path = self.checkpoint_dir / "jobs.json"
        self.cand_path = self.checkpoint_dir / "candidate_state.json"

        if isinstance(acquisition, str):
            self.acquisition = build_acquisition(acquisition, beta=beta)
        else:
            self.acquisition = acquisition

        self.n_obj = self.schema.n_expensive
        if self.n_obj < 1:
            raise ValueError("ObjectiveSchema must define at least one expensive objective")
        self.model: SurrogateModel = MultiOutputRFSurrogate(
            self.n_obj, n_estimators=n_estimators, seed=seed
        )
        self._target_round: int | None = None

        if self.use_plate_layout:
            self.plate.validate()

        if resume:
            if not (self.state_path.exists() and self.store_path.exists()):
                raise FileNotFoundError(
                    f"Cannot resume: missing checkpoint under {self.checkpoint_dir}"
                )
            self.state = CampaignState.load(self.state_path)
            self.store = ObservationStore.load(self.store_path)
            self.jobs = JobStore.load(self.jobs_path) if self.jobs_path.exists() else JobStore()
            self.cand = (
                CandidateStateStore.load(self.cand_path)
                if self.cand_path.exists()
                else CandidateStateStore(library.n)
            )
            self._reattach_open_jobs()
            log.info(
                "Resumed %s round=%s labeled=%s open_jobs=%s",
                self.state.campaign_id,
                self.state.round,
                len(self.store),
                len(self.jobs.open_jobs()),
            )
        else:
            if (
                self.state_path.exists()
                or self.store_path.exists()
                or self.jobs_path.exists()
            ):
                raise FileExistsError(
                    f"Checkpoint already exists at {self.checkpoint_dir}; "
                    "pass resume=True / --resume, or choose a new checkpoint_dir"
                )
            self.store = ObservationStore()
            self.jobs = JobStore()
            self.cand = CandidateStateStore(library.n)
            if self.constraints.stock_available is not None:
                self.cand.mark_stockout(~self.constraints.stock_available.astype(bool))
            self.state = CampaignState(
                campaign_id=campaign_id,
                seed=seed,
                round=0,
                positive_idx=list(positive_idx or []),
                negative_idx=list(negative_idx or []),
                init_done=False,
                schema=self.schema.to_dict(),
            )
            self._resolve_controls()
            self.cand.mark_control(self.state.positive_idx + self.state.negative_idx)
            self._submit_init()
            self._checkpoint()

    # ------------------------------------------------------------------ controls
    def _resolve_controls(self) -> None:
        """Controls come from config; never from hidden activity labels."""
        need_p = self.plate.n_positive if self.use_plate_layout else 0
        need_n = self.plate.n_negative if self.use_plate_layout else 0
        if need_p == 0 and need_n == 0:
            return
        rng = np.random.default_rng(self.state.seed)
        avail = self.cand.available_indices()
        if len(self.state.positive_idx) < need_p or len(self.state.negative_idx) < need_n:
            # Prefer static qed extremes when available; else random
            if self.library.static_Y is not None and "qed" in self.library.static_names:
                qed = self.library.static_col("qed")
                order = np.argsort(-qed)
                pos = [int(i) for i in order[:need_p]]
                neg = [int(i) for i in order[-need_n:]] if need_n else []
            else:
                pick = rng.choice(avail, size=min(len(avail), need_p + need_n), replace=False)
                pos = pick[:need_p].tolist()
                neg = pick[need_p : need_p + need_n].tolist()
            if len(self.state.positive_idx) < need_p:
                self.state.positive_idx = pos
            if len(self.state.negative_idx) < need_n:
                self.state.negative_idx = neg
            log.info(
                "Controls from static/random (not hidden labels): +%s -%s",
                self.state.positive_idx,
                self.state.negative_idx,
            )

    # ------------------------------------------------------------------ jobs
    def _submit_init(self) -> None:
        rng = np.random.default_rng(self.state.seed)
        avail = self.cand.available_indices()
        init_k = max(1, int(self.init_frac * self.library.n))
        init_k = min(init_k, len(avail))
        if self.use_plate_layout:
            init_k = min(init_k, self.plate.n_experimental)
        idxs = rng.choice(avail, size=init_k, replace=False).tolist()
        self._submit_batch(0, idxs, acquisition="init")

    def _reattach_open_jobs(self) -> None:
        for rec in self.jobs.open_jobs():
            try:
                self.executor.poll(rec.remote_job_id)
            except Exception:
                # In-process executors lose memory across processes — re-submit
                jid = self.executor.submit(rec.job)
                rec.remote_job_id = jid
                rec.retry_count += 1
                rec.message = "reattached via re-submit"
                log.warning("Re-submitted job %s as %s after resume", rec.job.job_id, jid)

    def _make_job(self, rnd: int, pool_indices: list[int], kind_all: ItemKind = ItemKind.EXPERIMENTAL) -> Job:
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
        job_id = f"{self.state.campaign_id}-r{rnd}-{uuid.uuid4().hex[:8]}"
        return Job(
            job_id=job_id,
            campaign_id=self.state.campaign_id,
            round=rnd,
            items=items,
            executor_kind=self.executor.kind,
            idempotency_key=job_id,
        )

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
                raise ValueError(f"Well index {w} exceeds plate capacity {self.plate.capacity}")
            s = f"{letters[w // self.plate.cols]}{w % self.plate.cols + 1:02d}"
            w += 1
            return s

        if len(exp_indices) > self.plate.n_experimental:
            log.warning(
                "Truncating experimental batch %d -> %d",
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

    def _submit_batch(self, rnd: int, idxs: list[int], acquisition: str) -> str:
        job = self._make_job(rnd, idxs)
        exp_idxs = [
            it.pool_idx
            for it in job.items
            if it.kind is ItemKind.EXPERIMENTAL and it.pool_idx >= 0
        ]
        self.cand.mark_selected(exp_idxs, job.job_id)
        remote_id = self.executor.submit(job)
        self.cand.mark_submitted(exp_idxs, remote_id)
        rec = JobRecord(
            job=job,
            remote_job_id=remote_id,
            status=JobLifecycle.SUBMITTED,
            submitted_at=time.time(),
        )
        self.jobs.put(rec)
        log.info(
            "Submitted job %s remote=%s n_exp=%d acquisition=%s",
            job.job_id,
            remote_id,
            len(exp_idxs),
            acquisition,
        )
        return remote_id

    def _poll_open(self) -> tuple[list[Observation], list[JobRecord]]:
        """Poll open jobs; return newly seen observations and newly completed jobs."""
        newly: list[Observation] = []
        completed: list[JobRecord] = []
        for rec in list(self.jobs.open_jobs()):
            try:
                status = self.executor.poll(rec.remote_job_id)
            except Exception as e:
                log.error("Poll failed for %s: %s — attempting re-submit", rec.job.job_id, e)
                try:
                    jid = self.executor.submit(rec.job)
                    rec.remote_job_id = jid
                    rec.retry_count += 1
                    status = self.executor.poll(jid)
                except Exception as e2:
                    rec.status = JobLifecycle.FAILED
                    rec.message = str(e2)
                    continue
            rec.last_poll_at = time.time()
            rec.status = JobLifecycle.RUNNING if not status.done else JobLifecycle.DONE
            for o in status.observations:
                key = o.item_id or f"{o.pool_idx}:{o.state.value}"
                if key in rec.seen_item_ids:
                    continue
                # Only ingest terminal-ish or usable progressive results
                if o.state in (
                    WellState.COMPLETED,
                    WellState.FAILED,
                    WellState.QC_REJECTED,
                    WellState.CANCELLED,
                ) or o.usable:
                    rec.seen_item_ids.add(key)
                    newly.append(o)
            if status.done:
                completed.append(rec)
        return newly, completed

    def _ingest(self, observations: list[Observation]) -> int:
        n_train = self.store.add_many(observations)
        self.cand.apply_observations(observations)
        return n_train

    def _acquisition_ref_point(self, labeled_Y: np.ndarray) -> np.ndarray:
        """Ref point from observed labels only — never from hidden oracle."""
        if labeled_Y.size == 0:
            return np.zeros(self.n_obj)
        return make_ref_point(labeled_Y)

    def _maybe_submit(self) -> bool:
        if len(self.jobs.open_jobs()) >= self.max_active_jobs:
            return False
        if len(self.store) == 0:
            return False
        inflight_acq = sum(1 for r in self.jobs.open_jobs() if r.job.round > 0)
        if self._target_round is not None and self.state.round + inflight_acq >= self._target_round:
            return False
        avail = self.cand.available_indices()
        if len(avail) == 0:
            return False

        X, Y = self.store.matrix(self.library.X, self.n_obj)
        self.model.fit(X, Y)
        means, stds = self.model.predict(self.library.X[avail])

        feas = self.constraints.feasible_mask(
            len(avail),
            pool_global_idx=avail,
        )
        feas_local = np.where(feas)[0]
        if len(feas_local) == 0:
            feas_local = np.arange(len(avail))

        k = self.plate.n_experimental if self.use_plate_layout else self.batch_size
        # free capacity in compounds: allow filling remaining active-job slots
        free_jobs = self.max_active_jobs - len(self.jobs.open_jobs())
        k = min(k, len(feas_local), max(1, free_jobs * k))

        ref = self._acquisition_ref_point(Y)
        rng = np.random.default_rng(self.state.seed * 1000 + self.state.round + 1 + len(self.jobs))
        sel = self.acquisition.select(
            avail[feas_local],
            means[feas_local],
            stds[feas_local],
            k,
            labeled_Y=Y,
            ref_point=ref,
            rng=rng,
        )
        chosen = list(sel.pool_indices)
        if self.use_plate_layout and sel.scores:
            score_map = {g: s for g, s in zip(sel.pool_indices, sel.scores)}
            cand = np.asarray(sel.pool_indices)
            scores = np.array([score_map[g] for g in cand])
            div_local = self.constraints.diversify(
                scores, self.library.X[cand], min(k, len(cand))
            )
            chosen = [int(cand[i]) for i in div_local]

        rnd = self.state.round + 1
        self._submit_batch(rnd, chosen, acquisition=self.acquisition.name)
        return True

    def _metrics_dict(self) -> dict[str, float]:
        if self.evaluator is None:
            return {"hv_frac": float("nan"), "pareto_recall": float("nan")}
        rep = self.evaluator.evaluate(self.store.labeled_indices)
        return {
            "hv_frac": rep.hv_frac,
            "pareto_recall": rep.pareto_recall,
            "mean_activity": rep.mean_activity,
        }

    def _checkpoint(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.store.save(self.store_path)
        self.state.save(self.state_path)
        self.jobs.save(self.jobs_path)
        self.cand.save(self.cand_path)

    def step(self) -> dict[str, Any]:
        """One non-blocking orchestration tick: poll → ingest → maybe submit."""
        newly, completed = self._poll_open()
        n_new_train = self._ingest(newly) if newly else 0

        for rec in completed:
            if not self.state.init_done:
                self.state.init_done = True
                m = self._metrics_dict()
                self.state.history.append(
                    {
                        "round": 0,
                        "n_labeled": len(self.store),
                        "hv_frac": m.get("hv_frac", float("nan")),
                        "pareto_recall": m.get("pareto_recall", float("nan")),
                        "submitted": len(rec.job.items),
                        "completed": sum(1 for o in newly if o.usable),
                        "failed": sum(1 for o in newly if o.state is WellState.FAILED),
                        "qc_rejected": sum(1 for o in newly if o.state is WellState.QC_REJECTED),
                        "job_id": rec.job.job_id,
                        "acquisition": "init",
                    }
                )
            elif rec.job.round > 0:
                self.state.round = max(self.state.round, rec.job.round)
                m = self._metrics_dict()
                self.state.history.append(
                    {
                        "round": rec.job.round,
                        "n_labeled": len(self.store),
                        "hv_frac": m.get("hv_frac", float("nan")),
                        "pareto_recall": m.get("pareto_recall", float("nan")),
                        "submitted": len(rec.job.items),
                        "completed": n_new_train,
                        "failed": sum(1 for o in newly if o.state is WellState.FAILED),
                        "qc_rejected": sum(1 for o in newly if o.state is WellState.QC_REJECTED),
                        "job_id": rec.job.job_id,
                        "acquisition": self.acquisition.name,
                    }
                )
                log.info(
                    "[%s] r%02d n=%d hv_frac=%s open=%d",
                    self.state.campaign_id,
                    self.state.round,
                    len(self.store),
                    m.get("hv_frac"),
                    len(self.jobs.open_jobs()),
                )

        submitted = False
        if self.state.init_done:
            submitted = self._maybe_submit()

        self._checkpoint()
        return {
            "round": self.state.round,
            "init_done": self.state.init_done,
            "n_labeled": len(self.store),
            "n_new_train": n_new_train,
            "n_new_obs": len(newly),
            "open_jobs": len(self.jobs.open_jobs()),
            "submitted": submitted,
            **self._metrics_dict(),
        }

    def run(self, n_rounds: int, *, max_steps: int | None = None) -> list[dict]:
        """Drive ``step`` until ``n_rounds`` acquisition rounds complete after init."""
        self._target_round = self.state.round + n_rounds
        limit = max_steps if max_steps is not None else max(2000, (n_rounds + 3) * self.max_polls * 10)
        for _ in range(limit):
            info = self.step()
            finished = (
                self.state.init_done
                and self.state.round >= self._target_round
                and not self.jobs.open_jobs()
            )
            exhausted = (
                self.state.init_done
                and not self.jobs.open_jobs()
                and len(self.cand.available_indices()) == 0
            )
            if finished or exhausted:
                return self.state.history
        raise TimeoutError(
            f"Campaign exceeded max_steps={limit} "
            f"(round={self.state.round}, open={info['open_jobs']}, labeled={info['n_labeled']})"
        )
