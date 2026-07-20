"""AutoDock Vina executor for docking-based virtual screening.

Requires the `vina` binary on PATH (or configured via vina_bin) and a receptor PDBQT.
When receptor is missing, construction still succeeds but submit raises a clear error
so unit tests can skip without failing the import path.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from autoscreen.core.types import ItemKind, Job, JobStatus, Observation, WellState
from autoscreen.executors.base import Executor
from autoscreen.logging_utils import get_logger

log = get_logger("vina")


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


@dataclass
class _JobRec:
    job: Job
    results: dict[str, Observation] = field(default_factory=dict)
    done: bool = False


class VinaExecutor(Executor):
    kind = "vina"

    def __init__(self, config: VinaConfig, qed_sa_lookup: dict[str, tuple[float, float]] | None = None):
        """qed_sa_lookup maps smiles -> (qed, sa_ease) in maximize convention for MOO."""
        self.config = config
        self.qed_sa_lookup = qed_sa_lookup or {}
        self._jobs: dict[str, _JobRec] = {}
        self._idempotency: dict[str, str] = {}
        Path(config.work_dir).mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return bool(self.config.receptor) and shutil.which(self.config.vina_bin) is not None

    def submit(self, job: Job) -> str:
        if not self.config.receptor:
            raise RuntimeError(
                "VinaExecutor: receptor PDBQT path is not configured. "
                "Set vina.receptor in the config file."
            )
        if shutil.which(self.config.vina_bin) is None:
            raise RuntimeError(
                f"VinaExecutor: binary '{self.config.vina_bin}' not found on PATH. "
                "Install AutoDock Vina or set vina.vina_bin."
            )
        key = job.idempotency_key or job.job_id
        if key in self._idempotency:
            return self._idempotency[key]
        job_id = job.job_id or f"vina-{uuid.uuid4().hex[:12]}"
        self._jobs[job_id] = _JobRec(job=job)
        self._idempotency[key] = job_id
        # Run docking synchronously on first poll for simplicity / reliability
        return job_id

    def _dock_one(self, smiles: str, work: Path) -> float | None:
        """Return docking score (lower is better) or None on failure."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as e:
            raise RuntimeError("RDKit required for VinaExecutor ligand prep") from e

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=0) != 0:
            return None
        AllChem.UFFOptimizeMolecule(mol)
        ligand_pdb = work / "ligand.pdb"
        Chem.MolToPDBFile(mol, str(ligand_pdb))

        # Prefer OpenBabel if present for PDB->PDBQT; otherwise skip with failure
        ligand_pdbqt = work / "ligand.pdbqt"
        obabel = shutil.which("obabel") or shutil.which("obabel.exe")
        if obabel:
            subprocess.run(
                [obabel, str(ligand_pdb), "-O", str(ligand_pdbqt)],
                check=False, capture_output=True,
            )
        if not ligand_pdbqt.exists():
            log.warning("Could not produce ligand PDBQT for %s (need OpenBabel)", smiles[:40])
            return None

        out = work / "out.pdbqt"
        log_path = work / "vina.log"
        cx, cy, cz = self.config.box_center
        sx, sy, sz = self.config.box_size
        cmd = [
            self.config.vina_bin,
            "--receptor", self.config.receptor,
            "--ligand", str(ligand_pdbqt),
            "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
            "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
            "--exhaustiveness", str(self.config.exhaustiveness),
            "--num_modes", str(self.config.num_modes),
            "--cpu", str(self.config.cpu),
            "--out", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        # Parse best affinity from stdout
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "1":
                try:
                    return float(parts[1])
                except ValueError:
                    continue
        return None

    def _run_job(self, rec: _JobRec) -> None:
        for it in rec.job.items:
            if it.kind is ItemKind.BLANK or it.pool_idx < 0:
                rec.results[it.item_id] = Observation(
                    smiles=it.smiles, pool_idx=it.pool_idx, values=None,
                    state=WellState.COMPLETED, qc_passed=False, source=self.kind,
                    item_id=it.item_id, kind=it.kind, message="blank",
                    timestamp=time.time(),
                )
                continue
            with tempfile.TemporaryDirectory(dir=self.config.work_dir) as td:
                score = self._dock_one(it.smiles, Path(td))
            if score is None:
                rec.results[it.item_id] = Observation(
                    smiles=it.smiles, pool_idx=it.pool_idx, values=None,
                    state=WellState.FAILED, qc_passed=False, source=self.kind,
                    item_id=it.item_id, kind=it.kind, message="vina failed",
                    timestamp=time.time(),
                )
                continue
            # Expensive objective only (activity = -dock). QED/SA are library static props.
            values = [-score]
            rec.results[it.item_id] = Observation(
                smiles=it.smiles, pool_idx=it.pool_idx, values=values,
                state=WellState.COMPLETED, qc_passed=True, source=self.kind,
                item_id=it.item_id, kind=it.kind, message="ok",
                raw={"dock": score}, timestamp=time.time(),
            )
        rec.done = True

    def poll(self, job_id: str) -> JobStatus:
        rec = self._jobs[job_id]
        if not rec.done:
            self._run_job(rec)
        return JobStatus(
            job_id=job_id,
            done=rec.done,
            observations=list(rec.results.values()),
            n_pending=0 if rec.done else len(rec.job.items),
            round=rec.job.round,
        )

    def cancel(self, job_id: str) -> None:
        if job_id in self._jobs and not self._jobs[job_id].done:
            self._jobs[job_id].done = True
