"""CampaignManager: async-capable active-learning orchestration."""
from __future__ import annotations

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
from autoscreen.core.persist import atomic_write_json
from autoscreen.core.types import ItemKind, Job, JobItem, Observation, WellState
from autoscreen.executors.base import Executor, JobNotFound
from autoscreen.logging_utils import get_logger

log = get_logger("campaign")

MAX_POLL_FAILS = 8
MAX_RESUBMIT = 3


@dataclass
class CampaignState:
    campaign_id: str
    seed: int
    round: int  # completed acquisition batches (analysis counter)
    history: list[dict] = field(default_factory=list)
    positive_idx: list[int] = field(default_factory=list)
    negative_idx: list[int] = field(default_factory=list)
    init_done: bool = False
    schema: dict = field(default_factory=dict)
    next_batch_seq: int = 0  # monotonic submit counter for unique IDs

    def save(self, path: Path) -> None:
        atomic_write_json(path, asdict(self))

    @classmethod
    def load(cls, path: Path) -> "CampaignState":
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("next_batch_seq", 0)
        return cls(**data)


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
        poll_interval_s: float = 0.0,
        max_wall_time_s: float | None = None,
        max_idle_time_s: float | None = None,
        pending_penalty: float = 0.5,
    ):
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
        self.poll_interval_s = float(poll_interval_s)
        self.max_wall_time_s = max_wall_time_s
        self.max_idle_time_s = max_idle_time_s
        self.pending_penalty = float(pending_penalty)
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
                "Resumed %s round=%s batch_seq=%s labeled=%s open_jobs=%s",
                self.state.campaign_id,
                self.state.round,
                self.state.next_batch_seq,
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
                next_batch_seq=0,
            )
            self._resolve_controls()
            self.cand.mark_control(self.state.positive_idx + self.state.negative_idx)
            self._submit_init()
            self._checkpoint()

    def _resolve_controls(self) -> None:
        """Controls must be explicit — never inferred from QED or hidden labels."""
        need_p = self.plate.n_positive if self.use_plate_layout else 0
        need_n = self.plate.n_negative if self.use_plate_layout else 0
        if need_p == 0 and need_n == 0:
            return
        if len(self.state.positive_idx) < need_p or len(self.state.negative_idx) < need_n:
            raise ValueError(
                "Plate layout requires explicit controls.positive_idx / negative_idx "
                f"(need +{need_p}/-{need_n}). AutoScreen will not infer biological "
                "controls from QED, SA, or hidden labels. Set n_positive/n_negative to 0 "
                "to disable plate controls."
            )

    def _submit_init(self) -> None:
        rng = np.random.default_rng(self.state.seed)
        avail = self.cand.available_indices()
        init_k = max(1, int(self.init_frac * self.library.n))
        init_k = min(init_k, len(avail))
        if self.use_plate_layout:
            init_k = min(init_k, self.plate.n_experimental)
        idxs = rng.choice(avail, size=init_k, replace=False).tolist()
        self._submit_batch(idxs, acquisition="init")

    def _experimental_idxs(self, job: Job) -> list[int]:
        return [
            it.pool_idx
            for it in job.items
            if it.kind in (ItemKind.EXPERIMENTAL, ItemKind.REPLICATE) and it.pool_idx >= 0
        ]

    def _reattach_open_jobs(self) -> None:
        for rec in self.jobs.open_jobs():
            if rec.status is JobLifecycle.PREPARED or not rec.remote_job_id:
                try:
                    remote = self.executor.submit(rec.job)
                    rec.remote_job_id = remote
                    rec.status = JobLifecycle.SUBMITTED
                    rec.retry_count += 1
                    rec.message = "submitted after resume from PREPARED"
                    self.cand.mark_submitted(self._experimental_idxs(rec.job), rec.job.job_id)
                    log.warning("Completed deferred submit for %s -> %s", rec.job.job_id, remote)
                except Exception as e:
                    rec.status = JobLifecycle.FAILED
                    rec.message = f"resume submit failed: {e}"
                    self.cand.release(self._experimental_idxs(rec.job), job_id=rec.job.job_id)
                continue
            try:
                self.executor.poll(rec.remote_job_id)
            except JobNotFound:
                if rec.retry_count >= MAX_RESUBMIT:
                    rec.status = JobLifecycle.FAILED
                    rec.message = "JobNotFound and max resubmit exceeded"
                    self.cand.release(self._experimental_idxs(rec.job), job_id=rec.job.job_id)
                    continue
                jid = self.executor.submit(rec.job)
                rec.remote_job_id = jid
                rec.retry_count += 1
                rec.message = "reattached via re-submit after JobNotFound"
                log.warning("Re-submitted job %s as %s after JobNotFound", rec.job.job_id, jid)
            except Exception as e:
                rec.poll_fail_count += 1
                rec.message = f"poll error on resume (will retry): {e}"
                log.warning("Poll error on resume for %s: %s", rec.job.job_id, e)
        self._checkpoint()

    def _alloc_batch_seq(self) -> int:
        seq = self.state.next_batch_seq
        self.state.next_batch_seq += 1
        return seq

    def _make_job(
        self,
        batch_seq: int,
        pool_indices: list[int],
        *,
        acquisition: str,
        kind_all: ItemKind = ItemKind.EXPERIMENTAL,
    ) -> Job:
        job_id = f"{self.state.campaign_id}-b{batch_seq}-{uuid.uuid4().hex[:8]}"
        if self.use_plate_layout:
            items = self._plate_items(job_id, batch_seq, pool_indices)
        else:
            items = [
                JobItem(
                    item_id=f"{job_id}:i{i}",
                    smiles=self.library.smis[gidx],
                    pool_idx=gidx,
                    kind=kind_all,
                    compound_id=f"CMP{gidx:07d}",
                )
                for i, gidx in enumerate(pool_indices)
            ]
        return Job(
            job_id=job_id,
            campaign_id=self.state.campaign_id,
            round=batch_seq,  # unique analytical batch id (not completion counter)
            items=items,
            executor_kind=self.executor.kind,
            idempotency_key=job_id,
            meta={"acquisition": acquisition, "batch_seq": batch_seq},
        )

    def _plate_items(self, job_id: str, batch_seq: int, exp_indices: list[int]) -> list[JobItem]:
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
                    item_id=f"{job_id}:{well}",
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
                    item_id=f"{job_id}:{well}",
                    well_id=well,
                    smiles=self.library.smis[gidx],
                    pool_idx=gidx,
                    kind=ItemKind.POSITIVE,
                    compound_id=f"CMP{gidx:07d}",
                )
            )
        for gidx in self.state.negative_idx[: self.plate.n_negative]:
            well = wid()
            items.append(
                JobItem(
                    item_id=f"{job_id}:{well}",
                    well_id=well,
                    smiles=self.library.smis[gidx],
                    pool_idx=gidx,
                    kind=ItemKind.NEGATIVE,
                    compound_id=f"CMP{gidx:07d}",
                )
            )
        for _ in range(self.plate.n_blank):
            well = wid()
            items.append(
                JobItem(
                    item_id=f"{job_id}:{well}",
                    well_id=well,
                    smiles="",
                    pool_idx=-1,
                    kind=ItemKind.BLANK,
                )
            )
        rng = np.random.default_rng(self.state.seed * 1000 + batch_seq)
        if exp_indices and self.plate.n_replicate > 0:
            n_rep = min(self.plate.n_replicate, len(exp_wells))
            picks = rng.choice(len(exp_wells), size=n_rep, replace=False)
            for j in picks:
                gidx = exp_indices[j]
                well = wid()
                items.append(
                    JobItem(
                        item_id=f"{job_id}:{well}",
                        well_id=well,
                        smiles=self.library.smis[gidx],
                        pool_idx=gidx,
                        kind=ItemKind.REPLICATE,
                        compound_id=f"CMP{gidx:07d}",
                        replicate_of=exp_wells[j],
                    )
                )
        return items

    def _submit_batch(self, idxs: list[int], acquisition: str) -> str:
        """Two-phase durable submit: PREPARE+checkpoint → remote → SUBMITTED+checkpoint."""
        batch_seq = self._alloc_batch_seq()
        job = self._make_job(batch_seq, idxs, acquisition=acquisition)
        exp_idxs = [
            it.pool_idx
            for it in job.items
            if it.kind is ItemKind.EXPERIMENTAL and it.pool_idx >= 0
        ]

        self.cand.mark_selected(exp_idxs, job.job_id)
        rec = JobRecord(
            job=job,
            remote_job_id="",
            status=JobLifecycle.PREPARED,
            submitted_at=time.time(),
            message=f"prepared acquisition={acquisition}",
        )
        self.jobs.put(rec)
        self._checkpoint()  # durable before remote call

        try:
            remote_id = self.executor.submit(job)
        except Exception as e:
            rec.status = JobLifecycle.FAILED
            rec.message = f"remote submit failed: {e}"
            self.cand.release(exp_idxs, job_id=job.job_id)
            self._checkpoint()
            raise

        rec.remote_job_id = remote_id
        rec.status = JobLifecycle.SUBMITTED
        self.cand.mark_submitted(exp_idxs, job.job_id)
        self._checkpoint()
        log.info(
            "Submitted job %s remote=%s batch_seq=%d n_exp=%d acquisition=%s",
            job.job_id,
            remote_id,
            batch_seq,
            len(exp_idxs),
            acquisition,
        )
        return remote_id

    def _poll_open(self) -> tuple[dict[str, list[Observation]], list[JobRecord], float]:
        """Poll open jobs; return newly seen observations, completed jobs, sleep hint."""
        newly_by_job: dict[str, list[Observation]] = {}
        completed: list[JobRecord] = []
        sleep_hint = 0.0

        for rec in list(self.jobs.open_jobs()):
            if rec.status is JobLifecycle.PREPARED or not rec.remote_job_id:
                try:
                    remote = self.executor.submit(rec.job)
                    rec.remote_job_id = remote
                    rec.status = JobLifecycle.SUBMITTED
                    self.cand.mark_submitted(self._experimental_idxs(rec.job), rec.job.job_id)
                    self._checkpoint()
                except Exception as e:
                    rec.status = JobLifecycle.FAILED
                    rec.message = f"deferred submit failed: {e}"
                    self.cand.release(self._experimental_idxs(rec.job), job_id=rec.job.job_id)
                continue

            try:
                status = self.executor.poll(rec.remote_job_id)
                rec.poll_fail_count = 0
                sleep_hint = max(sleep_hint, float(getattr(status, "next_poll_after", 0.0) or 0.0))
            except JobNotFound as e:
                if rec.retry_count >= MAX_RESUBMIT:
                    rec.status = JobLifecycle.FAILED
                    rec.message = f"JobNotFound max resubmit: {e}"
                    self.cand.release(self._experimental_idxs(rec.job), job_id=rec.job.job_id)
                    continue
                try:
                    jid = self.executor.submit(rec.job)
                    rec.remote_job_id = jid
                    rec.retry_count += 1
                    rec.message = "re-submit after JobNotFound"
                    status = self.executor.poll(jid)
                except Exception as e2:
                    rec.status = JobLifecycle.FAILED
                    rec.message = str(e2)
                    self.cand.release(self._experimental_idxs(rec.job), job_id=rec.job.job_id)
                    continue
            except Exception as e:
                rec.poll_fail_count += 1
                rec.message = f"poll error ({rec.poll_fail_count}): {e}"
                log.warning("Poll failed for %s (no resubmit): %s", rec.job.job_id, e)
                if rec.poll_fail_count >= MAX_POLL_FAILS:
                    rec.status = JobLifecycle.FAILED
                    rec.message = f"max poll failures: {e}"
                    self.cand.release(self._experimental_idxs(rec.job), job_id=rec.job.job_id)
                sleep_hint = max(sleep_hint, self.poll_interval_s or 0.5)
                continue

            rec.last_poll_at = time.time()
            rec.status = JobLifecycle.RUNNING if not status.done else JobLifecycle.DONE

            running_idxs = [
                o.pool_idx
                for o in status.observations
                if o.state is WellState.RUNNING and o.pool_idx >= 0
            ]
            if running_idxs:
                self.cand.mark_running(running_idxs)
            if not status.done:
                self.cand.mark_running(
                    [
                        it.pool_idx
                        for it in rec.job.items
                        if it.kind is ItemKind.EXPERIMENTAL and it.pool_idx >= 0
                    ]
                )

            job_new: list[Observation] = []
            for o in status.observations:
                key = o.item_id or f"{o.pool_idx}:{o.state.value}"
                if key in rec.seen_item_ids:
                    continue
                if o.state in (
                    WellState.COMPLETED,
                    WellState.FAILED,
                    WellState.QC_REJECTED,
                    WellState.CANCELLED,
                ) or o.usable or o.contributes_measurement:
                    rec.seen_item_ids.add(key)
                    job_new.append(o)
            if job_new:
                newly_by_job[rec.job.job_id] = job_new
            if status.done:
                completed.append(rec)

        return newly_by_job, completed, sleep_hint

    def _ingest(self, observations: list[Observation]) -> int:
        n_train = self.store.add_many(observations)
        self.cand.apply_observations(observations)
        self.cand.sync_from_store(
            self.store.labeled_indices,
            aggregate_qc_rejected=self.store._aggregate_qc_rejected,
        )
        return n_train

    def _acquisition_ref_point(self, labeled_Y: np.ndarray) -> np.ndarray:
        if labeled_Y.size == 0:
            return np.zeros(self.n_obj)
        return make_ref_point(labeled_Y)

    def _is_acquisition_job(self, rec: JobRecord) -> bool:
        return rec.job.meta.get("acquisition", "") != "init"

    def _pending_fps(self, open_jobs: list[JobRecord]) -> np.ndarray | None:
        idxs: list[int] = []
        for rec in open_jobs:
            for it in rec.job.items:
                if it.kind is ItemKind.EXPERIMENTAL and it.pool_idx >= 0:
                    idxs.append(int(it.pool_idx))
        if not idxs:
            return None
        return self.library.X[np.asarray(sorted(set(idxs)), dtype=int)]

    def _maybe_submit(self) -> bool:
        open_jobs = self.jobs.open_jobs()
        if len(open_jobs) >= self.max_active_jobs:
            return False
        if len(self.store) == 0:
            return False
        inflight_acq = sum(1 for r in open_jobs if self._is_acquisition_job(r))
        if self._target_round is not None and self.state.round + inflight_acq >= self._target_round:
            return False
        avail = self.cand.available_indices()
        if len(avail) == 0:
            return False

        X, Y = self.store.matrix(self.library.X, self.n_obj)
        self.model.fit(X, Y)
        means, stds = self.model.predict(self.library.X[avail])

        feas = self.constraints.feasible_mask(len(avail), pool_global_idx=avail)
        feas_local = np.where(feas)[0]
        if len(feas_local) == 0:
            feas_local = np.arange(len(avail))

        k = self.plate.n_experimental if self.use_plate_layout else self.batch_size
        free_jobs = self.max_active_jobs - len(open_jobs)
        k = min(k, len(feas_local), max(1, free_jobs * k))

        pool = avail[feas_local]
        ref = self._acquisition_ref_point(Y)
        rng = np.random.default_rng(
            self.state.seed * 1000 + self.state.next_batch_seq + len(self.jobs)
        )
        pending_fps = self._pending_fps(open_jobs)
        sel = self.acquisition.select(
            pool,
            means[feas_local],
            stds[feas_local],
            k,
            labeled_Y=Y,
            ref_point=ref,
            rng=rng,
            cand_fps=self.library.X[pool],
            pending_fps=pending_fps,
            pending_penalty=self.pending_penalty,
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

        self._submit_batch(chosen, acquisition=self.acquisition.name)
        return True

    def _metrics_dict(self) -> dict[str, float]:
        if self.evaluator is None:
            return {
                "top1_recall": float("nan"),
                "ef_top1": float("nan"),
                "bedroc": float("nan"),
            }
        rep = self.evaluator.evaluate(self.store.labeled_indices)
        return {
            "top01_recall": rep.top01_recall,
            "top1_recall": rep.top1_recall,
            "ef_top1": rep.ef_top1,
            "bedroc": rep.bedroc,
            "n_hits_top1": float(rep.n_hits_top1),
            "mean_activity": rep.mean_activity,
            "hv_frac": rep.hv_frac,
            "pareto_recall": rep.pareto_recall,
        }

    def _job_history_stats(self, rec: JobRecord) -> dict[str, int]:
        item_ids = {it.item_id for it in rec.job.items}
        obs = [o for o in self.store.history if o.item_id in item_ids]
        return {
            "completed": sum(1 for o in obs if o.contributes_measurement),
            "failed": sum(1 for o in obs if o.state is WellState.FAILED),
            "qc_rejected": sum(1 for o in obs if o.state is WellState.QC_REJECTED),
        }

    def _checkpoint(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.store.save(self.store_path)
        self.state.save(self.state_path)
        self.jobs.save(self.jobs_path)
        self.cand.save(self.cand_path)

    def step(self) -> dict[str, Any]:
        """One non-blocking orchestration tick: poll → ingest → maybe submit."""
        newly_by_job, completed, sleep_hint = self._poll_open()
        all_new = [o for obs in newly_by_job.values() for o in obs]
        n_new_train = self._ingest(all_new) if all_new else 0

        for rec in completed:
            stats = self._job_history_stats(rec)
            m = self._metrics_dict()
            acq = rec.job.meta.get("acquisition", "")
            if acq == "init" and not self.state.init_done:
                self.state.init_done = True
                self.state.history.append(
                    {
                        "round": 0,
                        "batch_seq": rec.job.meta.get("batch_seq", rec.job.round),
                        "n_labeled": len(self.store),
                        "top1_recall": m.get("top1_recall", float("nan")),
                        "ef_top1": m.get("ef_top1", float("nan")),
                        "bedroc": m.get("bedroc", float("nan")),
                        "hv_frac": m.get("hv_frac", float("nan")),
                        "pareto_recall": m.get("pareto_recall", float("nan")),
                        "submitted": len(rec.job.items),
                        "completed": stats["completed"],
                        "failed": stats["failed"],
                        "qc_rejected": stats["qc_rejected"],
                        "job_id": rec.job.job_id,
                        "acquisition": "init",
                    }
                )
            elif acq != "init":
                self.state.round += 1
                self.state.history.append(
                    {
                        "round": self.state.round,
                        "batch_seq": rec.job.meta.get("batch_seq", rec.job.round),
                        "n_labeled": len(self.store),
                        "top1_recall": m.get("top1_recall", float("nan")),
                        "ef_top1": m.get("ef_top1", float("nan")),
                        "bedroc": m.get("bedroc", float("nan")),
                        "hv_frac": m.get("hv_frac", float("nan")),
                        "pareto_recall": m.get("pareto_recall", float("nan")),
                        "submitted": len(rec.job.items),
                        "completed": stats["completed"],
                        "failed": stats["failed"],
                        "qc_rejected": stats["qc_rejected"],
                        "job_id": rec.job.job_id,
                        "acquisition": acq or self.acquisition.name,
                    }
                )
                log.info(
                    "[%s] r%02d batch=%s n=%d top1=%.3f EF=%.2f bedroc=%.3f open=%d",
                    self.state.campaign_id,
                    self.state.round,
                    rec.job.meta.get("batch_seq"),
                    len(self.store),
                    m.get("top1_recall") or 0.0,
                    m.get("ef_top1") or 0.0,
                    m.get("bedroc") or 0.0,
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
            "n_new_obs": len(all_new),
            "open_jobs": len(self.jobs.open_jobs()),
            "submitted": submitted,
            "next_batch_seq": self.state.next_batch_seq,
            "sleep_hint": sleep_hint,
            **self._metrics_dict(),
        }

    def run(
        self,
        n_rounds: int,
        *,
        max_steps: int | None = None,
        poll_interval_s: float | None = None,
        max_wall_time_s: float | None = None,
        max_idle_time_s: float | None = None,
    ) -> list[dict]:
        """Drive ``step`` until acquisition rounds complete (time-aware, not busy-spin)."""
        self._target_round = self.state.round + n_rounds
        interval = self.poll_interval_s if poll_interval_s is None else float(poll_interval_s)
        wall = self.max_wall_time_s if max_wall_time_s is None else max_wall_time_s
        idle_lim = self.max_idle_time_s if max_idle_time_s is None else max_idle_time_s
        # Soft step cap only as a safety net; wall/idle clocks are primary for Vina.
        limit = max_steps if max_steps is not None else max(50_000, (n_rounds + 3) * 5000)
        t0 = time.time()
        last_progress = t0
        info: dict[str, Any] = {}
        for step_i in range(limit):
            info = self.step()
            progressed = bool(info.get("n_new_obs") or info.get("submitted") or info.get("n_new_train"))
            if progressed:
                last_progress = time.time()
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
            now = time.time()
            if wall is not None and (now - t0) > float(wall):
                raise TimeoutError(
                    f"Campaign exceeded max_wall_time_s={wall} "
                    f"(round={self.state.round}, open={info.get('open_jobs')})"
                )
            if idle_lim is not None and (now - last_progress) > float(idle_lim):
                raise TimeoutError(
                    f"Campaign exceeded max_idle_time_s={idle_lim} "
                    f"(round={self.state.round}, open={info.get('open_jobs')})"
                )
            if info.get("open_jobs", 0) > 0 and not progressed:
                sleep_s = float(info.get("sleep_hint") or 0.0)
                if sleep_s <= 0:
                    sleep_s = interval if interval > 0 else (0.05 if self.executor.kind == "vina" else 0.0)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        raise TimeoutError(
            f"Campaign exceeded max_steps={limit} "
            f"(round={self.state.round}, open={info.get('open_jobs')}, labeled={info.get('n_labeled')})"
        )
