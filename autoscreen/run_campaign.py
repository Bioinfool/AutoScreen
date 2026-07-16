"""Run an asynchronous, plate-based active-learning campaign (mock robot).

Example:
    python -m autoscreen.run_campaign --rounds 6 --out runs/demo
    # interrupt, then resume from the checkpoint:
    python -m autoscreen.run_campaign --rounds 4 --out runs/demo --resume
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .campaign import Campaign
from .data import load_moo_dataset
from .mock_backend import MockRobotBackend
from .plate import PlateConfig
from .robot_client import MockRobotClient


def main() -> None:
    p = argparse.ArgumentParser(description="Async plate-based AL campaign (mock robot)")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/demo")
    p.add_argument("--campaign-id", type=str, default="screen_001")
    p.add_argument("--resume", action="store_true", help="continue from existing checkpoint")
    p.add_argument("--n-experimental", type=int, default=80)
    p.add_argument("--diversity-lambda", type=float, default=0.4)
    p.add_argument("--fail-rate", type=float, default=0.05)
    p.add_argument("--qc-reject-rate", type=float, default=0.05)
    args = p.parse_args()

    out_dir = Path(args.out)
    checkpoint = out_dir / "campaign_state.json"
    if not args.resume and checkpoint.exists():
        checkpoint.unlink()

    ds = load_moo_dataset()
    backend = MockRobotBackend(
        truth=ds.Y,
        fail_rate=args.fail_rate,
        qc_reject_rate=args.qc_reject_rate,
        seed=args.seed,
    )
    client = MockRobotClient(backend)
    plate_config = PlateConfig(
        n_experimental=args.n_experimental,
        diversity_lambda=args.diversity_lambda,
    )

    campaign = Campaign(
        ds=ds,
        client=client,
        plate_config=plate_config,
        campaign_id=args.campaign_id,
        seed=args.seed,
        checkpoint=checkpoint,
    )
    campaign.run(args.rounds)
    print(f"\nCheckpoint: {checkpoint}")
    print(f"Rounds completed: {campaign.state.round}, labeled: {len(campaign.state.labeled_idx)}")


if __name__ == "__main__":
    main()
