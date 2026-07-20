"""AutoDock Vina executor with per-ligand async tasks (non-blocking poll)."""
from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from autoscreen.core.types import ItemKind, Job, JobItem, JobStatus, Observation, WellState
from autoscreen.executors.base import Executor, JobNotFound
from autoscreen.logging_utils import get_logger

log = get_logger("vina")


class LigandTaskState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass
class VinaConfig:
    receptor: str | None = None
    box_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    box_size: tuple[float, float, float] = (20.0, 20.0, 20.0)
    exhaustiveness: int = 8
    num_modes: int = 1
    cpu: int = 4
    vina_bin: str = "vina"
    work_dir: str = "runs/vina_work"
    max_workers: int = 2
    per_ligand_timeout_s: float = 120.0
    max_retries: int = 1


@dataclass
class _LigandTask:
    item: JobItem
    state: LigandTaskState = LigandTaskState.QUEUED
    future: Future | None = None
    attempts: int = 0
    observation: Observation | None = None
    work_dir: Path | None = None


@dataclass
class _JobRec:
    job: Job
    tasks: dict[str, _LigandTask] = field(default_factory=dict)


def _parse_vina_score(stdout: str) -> float | None:
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "1":
            try:
                return float(parts[1])
            except ValueError:
                continue
    return None


def affinity_to_activity(affinity: float) -> float:
    """Map Vina affinity (kcal/mol, lower/better) to AutoScreen maximize activity."""
    return -float(affinity)


def _dock_ligand(
    *,
    smiles: str,
    work: Path,
    config: VinaConfig,
) -> tuple[float | None, str]:
    """Run one docking; returns (score_or_None, message). Persistent under ``work``."""
    work.mkdir(parents=True, exist_ok=True)
    score_path = work / "score.txt"
    if score_path.exists():
        try:
            return float(score_path.read_text(encoding="utf-8").strip()), "cached"
        except ValueError:
            pass

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:
        raise RuntimeError("RDKit required for VinaExecutor ligand prep") from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid smiles"
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0) != 0:
        return None, "embed failed"
    AllChem.UFFOptimizeMolecule(mol)
    ligand_pdb = work / "ligand.pdb"
    Chem.MolToPDBFile(mol, str(ligand_pdb))

    ligand_pdbqt = work / "ligand.pdbqt"
    obabel = shutil.which("obabel") or shutil.which("obabel.exe")
    if not obabel:
        return None, "OpenBabel not found on PATH (install openbabel-wheel or set PATH)"
    proc_ob = subprocess.run(
        [obabel, str(ligand_pdb), "-O", str(ligand_pdbqt)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc_ob.returncode != 0 or not ligand_pdbqt.exists():
        err = (proc_ob.stderr or proc_ob.stdout or "").strip()
        detail = err[:500] if err else f"exit {proc_ob.returncode}, no pdbqt written"
        return None, f"OpenBabel PDBQT conversion failed: {detail}"

    out = work / "out.pdbqt"
    log_path = work / "vina.log"
    cx, cy, cz = config.box_center
    sx, sy, sz = config.box_size
    assert config.receptor is not None
    cmd = [
        config.vina_bin,
        "--receptor",
        config.receptor,
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        str(cx),
        "--center_y",
        str(cy),
        "--center_z",
        str(cz),
        "--size_x",
        str(sx),
        "--size_y",
        str(sy),
        "--size_z",
        str(sz),
        "--exhaustiveness",
        str(config.exhaustiveness),
        "--num_modes",
        str(config.num_modes),
        "--cpu",
        str(config.cpu),
        "--out",
        str(out),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.per_ligand_timeout_s,
        )
    except subprocess.TimeoutExpired:
        (work / "timeout").write_text("1", encoding="utf-8")
        return None, "timeout"
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        return None, f"vina exit {proc.returncode}"
    score = _parse_vina_score(proc.stdout)
    if score is None:
        return None, "parse failed"
    # Cache raw Vina affinity (kcal/mol); Observation converts via affinity_to_activity.
    score_path.write_text(str(score), encoding="utf-8")
    return score, "ok"


class VinaExecutor(Executor):
    """Per-ligand thread pool; ``poll`` returns immediately with partial results."""

    kind = "vina"

    def __init__(
        self,
        config: VinaConfig,
        *,
        dock_fn=None,
    ):
        self.config = config
        self._dock_fn = dock_fn or _dock_ligand
        self._jobs: dict[str, _JobRec] = {}
        self._idempotency: dict[str, str] = {}
        self._rr = 0
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(config.max_workers)))
        Path(config.work_dir).mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        if not self.config.receptor:
            return False
        bin_path = Path(self.config.vina_bin)
        return bin_path.is_file() or shutil.which(self.config.vina_bin) is not None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def submit(self, job: Job) -> str:
        if not self.config.receptor:
            raise RuntimeError(
                "VinaExecutor: receptor PDBQT path is not configured. "
                "Set vina.receptor in the config file."
            )
        if self._dock_fn is _dock_ligand:
            bin_path = Path(self.config.vina_bin)
            if not (bin_path.is_file() or shutil.which(self.config.vina_bin)):
                raise RuntimeError(
                    f"VinaExecutor: binary '{self.config.vina_bin}' not found. "
                    "Install AutoDock Vina or set vina.vina_bin."
                )
        key = job.idempotency_key or job.job_id
        if key in self._idempotency:
            return self._idempotency[key]
        job_id = job.job_id or f"vina-{uuid.uuid4().hex[:12]}"
        tasks: dict[str, _LigandTask] = {}
        for it in job.items:
            work = Path(self.config.work_dir) / job_id / it.item_id.replace(":", "_")
            tasks[it.item_id] = _LigandTask(item=it, work_dir=work)
        self._jobs[job_id] = _JobRec(job=job, tasks=tasks)
        self._idempotency[key] = job_id
        return job_id

    def _finalize_task(self, task: _LigandTask, score: float | None, message: str) -> None:
        it = task.item
        common = dict(
            smiles=it.smiles,
            pool_idx=it.pool_idx,
            source=self.kind,
            compound_id=it.compound_id,
            item_id=it.item_id,
            kind=it.kind,
            timestamp=time.time(),
        )
        if it.kind is ItemKind.BLANK or it.pool_idx < 0:
            task.state = LigandTaskState.COMPLETED
            task.observation = Observation(
                values=None,
                state=WellState.COMPLETED,
                qc_passed=False,
                message="blank",
                **common,
            )
            return
        if message == "timeout":
            task.state = LigandTaskState.TIMEOUT
            task.observation = Observation(
                values=None,
                state=WellState.FAILED,
                qc_passed=False,
                message="vina timeout",
                **common,
            )
            return
        if score is None:
            task.state = LigandTaskState.FAILED
            task.observation = Observation(
                values=None,
                state=WellState.FAILED,
                qc_passed=False,
                message=message or "vina failed",
                **common,
            )
            return
        task.state = LigandTaskState.COMPLETED
        # Observation.values use maximize convention (stronger binder → larger activity).
        task.observation = Observation(
            values=[affinity_to_activity(score)],
            state=WellState.COMPLETED,
            qc_passed=True,
            message="ok",
            raw={"vina_affinity": float(score), "activity": affinity_to_activity(score)},
            **common,
        )

    def _start_task(self, job_id: str, task: _LigandTask) -> None:
        it = task.item
        if it.kind is ItemKind.BLANK or it.pool_idx < 0:
            self._finalize_task(task, None, "blank")
            return
        assert task.work_dir is not None
        task.attempts += 1
        task.state = LigandTaskState.RUNNING
        cfg = self.config

        def _run():
            return self._dock_fn(smiles=it.smiles, work=task.work_dir, config=cfg)

        task.future = self._pool.submit(_run)

    def _collect_finished(self, rec: _JobRec) -> None:
        for task in rec.tasks.values():
            if task.state is not LigandTaskState.RUNNING or task.future is None:
                continue
            if not task.future.done():
                continue
            try:
                score, message = task.future.result()
            except Exception as e:
                score, message = None, str(e)
            task.future = None
            if (
                score is None
                and message != "timeout"
                and task.attempts <= self.config.max_retries
            ):
                task.state = LigandTaskState.QUEUED
                continue
            self._finalize_task(task, score, message)

    def _fair_launch(self) -> None:
        """Round-robin launch across jobs so later jobs are not starved."""
        running = sum(
            1
            for r in self._jobs.values()
            for t in r.tasks.values()
            if t.state is LigandTaskState.RUNNING
        )
        capacity = max(0, self.config.max_workers - running)
        if capacity <= 0:
            return
        job_ids = list(self._jobs.keys())
        if not job_ids:
            return
        # Rotate start index each call for fairness
        start = getattr(self, "_rr", 0) % len(job_ids)
        self._rr = start + 1
        launched = 0
        for offset in range(len(job_ids)):
            if launched >= capacity:
                break
            jid = job_ids[(start + offset) % len(job_ids)]
            rec = self._jobs[jid]
            for task in rec.tasks.values():
                if launched >= capacity:
                    break
                if task.state is LigandTaskState.QUEUED:
                    self._start_task(jid, task)
                    if task.state is LigandTaskState.RUNNING:
                        launched += 1

    def poll(self, job_id: str) -> JobStatus:
        if job_id not in self._jobs:
            raise JobNotFound(job_id)
        # Always progress the global queue, not only this job's queued ligands
        for rec in self._jobs.values():
            self._collect_finished(rec)
        self._fair_launch()
        rec = self._jobs[job_id]

        observations: list[Observation] = []
        pending = 0
        for task in rec.tasks.values():
            if task.observation is not None:
                observations.append(task.observation)
            elif task.state is LigandTaskState.RUNNING:
                pending += 1
                observations.append(
                    Observation(
                        smiles=task.item.smiles,
                        pool_idx=task.item.pool_idx,
                        values=None,
                        state=WellState.RUNNING,
                        qc_passed=False,
                        source=self.kind,
                        item_id=task.item.item_id,
                        kind=task.item.kind,
                        message="running",
                        timestamp=time.time(),
                    )
                )
            else:
                pending += 1
        return JobStatus(
            job_id=job_id,
            done=pending == 0 and all(t.observation is not None for t in rec.tasks.values()),
            observations=observations,
            n_pending=pending,
            round=rec.job.round,
            next_poll_after=0.05 if pending else 0.0,
        )

    def cancel(self, job_id: str) -> None:
        if job_id not in self._jobs:
            return
        rec = self._jobs[job_id]
        for task in rec.tasks.values():
            if task.observation is not None:
                continue
            if task.future is not None and not task.future.done():
                task.future.cancel()
            task.state = LigandTaskState.FAILED
            task.observation = Observation(
                smiles=task.item.smiles,
                pool_idx=task.item.pool_idx,
                values=None,
                state=WellState.CANCELLED,
                qc_passed=False,
                source=self.kind,
                item_id=task.item.item_id,
                kind=task.item.kind,
                message="cancelled",
                timestamp=time.time(),
            )
