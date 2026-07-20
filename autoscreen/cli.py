"""Unified CLI: autoscreen run --config ..."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoscreen.config import load_config, project_root
from autoscreen.core.campaign import CampaignManager
from autoscreen.core.constraints import ConstraintManager, PlateConfig
from autoscreen.core.library import load_candidate_library
from autoscreen.executors.replay import ReplayExecutor
from autoscreen.executors.robot import RobotExecutor
from autoscreen.executors.vina import VinaConfig, VinaExecutor
from autoscreen.logging_utils import setup_logging, get_logger

log = get_logger("cli")


def _resolve(path: str | None, root: Path) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p


def build_from_config(cfg: dict, *, resume: bool = False) -> CampaignManager:
    root = project_root()
    data = cfg.get("data", {})
    library = load_candidate_library(
        _resolve(data["library_csv"], root),
        _resolve(data["fps_h5"], root),
        _resolve(data.get("moo_csv"), root),
    )

    plate_cfg = cfg.get("plate", {})
    plate = PlateConfig(
        n_experimental=int(plate_cfg.get("n_experimental", cfg.get("batch_size", 80))),
        n_positive=int(plate_cfg.get("n_positive", 4)),
        n_negative=int(plate_cfg.get("n_negative", 4)),
        n_blank=int(plate_cfg.get("n_blank", 4)),
        n_replicate=int(plate_cfg.get("n_replicate", 4)),
        diversity_lambda=float(plate_cfg.get("diversity_lambda", 0.4)),
        sa_feasibility_quantile=float(plate_cfg.get("sa_feasibility_quantile", 0.1)),
        rows=int(plate_cfg.get("rows", 8)),
        cols=int(plate_cfg.get("cols", 12)),
    )

    executor_kind = cfg.get("executor", "replay").lower()
    seed = int(cfg.get("seed", 0))

    # Stock-outs only for robot (or when explicitly configured).
    constraints_cfg = cfg.get("constraints", {})
    stock_rate = constraints_cfg.get("stock_out_rate")
    if stock_rate is None and executor_kind == "robot":
        stock_rate = 0.05
    stock = None
    if stock_rate is not None and float(stock_rate) > 0:
        stock_rng = np.random.default_rng(seed)
        stock = stock_rng.random(library.n) >= float(stock_rate)
        log.info("Applied stock_out_rate=%.3f (%d available)", float(stock_rate), int(stock.sum()))
    constraints = ConstraintManager(plate=plate, stock_available=stock)

    if executor_kind == "replay":
        executor = ReplayExecutor(
            library,
            activity_noise=float(cfg.get("activity_noise", 0.0)),
            fail_rate=float(cfg.get("fail_rate", 0.0)),
            qc_reject_rate=float(cfg.get("qc_reject_rate", 0.0)),
            seed=seed,
        )
        use_plate = False
        max_polls = 20
    elif executor_kind == "vina":
        v = cfg.get("vina", {})
        qed_sa = {}
        if library.Y_hidden is not None:
            for i, smi in enumerate(library.smis):
                qed_sa[smi] = (float(library.Y_hidden[i, 1]), float(library.Y_hidden[i, 2]))
        executor = VinaExecutor(
            VinaConfig(
                receptor=v.get("receptor"),
                box_center=tuple(v.get("box_center", [0, 0, 0])),
                box_size=tuple(v.get("box_size", [20, 20, 20])),
                exhaustiveness=int(v.get("exhaustiveness", 4)),
                num_modes=int(v.get("num_modes", 1)),
                cpu=int(v.get("cpu", 4)),
                vina_bin=v.get("vina_bin", "vina"),
                work_dir=str(_resolve(v.get("work_dir", "runs/vina_work"), root)),
            ),
            qed_sa_lookup=qed_sa,
        )
        use_plate = False
        max_polls = 5
    elif executor_kind == "robot":
        r = cfg.get("robot", {})
        import os
        base = os.environ.get("AUTOSCREEN_ROBOT_URL", r.get("base_url", "http://127.0.0.1:8080"))
        executor = RobotExecutor(
            base_url=base,
            timeout_s=float(r.get("timeout_s", 30)),
            poll_interval_s=float(r.get("poll_interval_s", 0.2)),
        )
        use_plate = True
        max_polls = int(r.get("max_polls", 100))
        truth = r.get("truth_moo") or data.get("moo_csv")
        if truth:
            log.info(
                "Robot campaigns require robot_mock truth aligned with moo_csv=%s "
                "(set AUTOSCREEN_TRUTH_MOO on the mock service)",
                truth,
            )
    else:
        raise ValueError(f"Unknown executor: {executor_kind}")

    ckpt = _resolve(cfg.get("checkpoint_dir", "runs/default"), root)
    sur = cfg.get("surrogate", {})
    return CampaignManager(
        library=library,
        executor=executor,
        acquisition=cfg.get("acquisition", "greedy"),
        campaign_id=cfg.get("campaign_id", "campaign"),
        seed=seed,
        batch_size=int(cfg.get("batch_size", 100)),
        init_frac=float(cfg.get("init_frac", 0.01)),
        beta=float(cfg.get("beta", 1.0)),
        checkpoint_dir=ckpt,
        n_estimators=int(sur.get("n_estimators", 100)),
        plate=plate,
        constraints=constraints,
        use_plate_layout=use_plate,
        max_polls=max_polls,
        resume=resume,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="AutoScreen active-learning campaign runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a campaign from a YAML config")
    run_p.add_argument("--config", required=True)
    run_p.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of additional acquisition rounds to run from the current state",
    )
    run_p.add_argument("--resume", action="store_true", help="Resume from checkpoint_dir")
    run_p.add_argument("--log-level", default="INFO")

    args = p.parse_args(argv)
    if args.cmd == "run":
        setup_logging(args.log_level)
        cfg = load_config(args.config)
        camp = build_from_config(cfg, resume=args.resume)
        n_rounds = args.rounds if args.rounds is not None else int(cfg.get("n_rounds", 5))
        history = camp.run(n_rounds)
        out = Path(camp.checkpoint_dir) / "history.json"
        out.write_text(json.dumps(history, indent=2), encoding="utf-8")
        log.info("Done. checkpoint=%s labeled=%s", camp.checkpoint_dir, len(camp.store))


if __name__ == "__main__":
    main()
