"""Dummy trainer: emits realistic events without doing any ML.

Used to validate the runner infrastructure (launch, watch, stop, replay, list)
before touching networks or self-play. Each 'iteration' sleeps a bit and
emits selfplay/train/eval/checkpoint events with plausible fake numbers.

Run via: python -m mlfactory.runner.dummy_trainer --run-dir ... --iters 10
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

from mlfactory.runner.events import write_event
from mlfactory.runner.layout import RunLayout


# --- Graceful stop ---------------------------------------------------------

_stop_requested = False


def _handle_sigterm(signum: int, frame: object) -> None:  # noqa: ARG001
    global _stop_requested
    _stop_requested = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dummy trainer for runner validation")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--iter-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    run_dir = args.run_dir
    # Reconstruct a layout from the run_dir for path helpers.
    run_id = run_dir.name
    game = run_dir.parent.name
    root = run_dir.parent.parent.parent
    layout = RunLayout(root=root, game=game, run_id=run_id)
    layout.ensure()

    rng = random.Random(args.seed)
    start = time.monotonic()
    layout.write_status("running")
    write_event(
        layout.events_path,
        "run_start",
        trainer="dummy",
        iters=args.iters,
    )

    fake_elo = 1500.0
    try:
        for i in range(1, args.iters + 1):
            if _stop_requested:
                break
            iter_start = time.monotonic()
            write_event(layout.events_path, "iter_start", iter=i)

            # Fake self-play
            time.sleep(args.iter_seconds * 0.4)
            avg_moves = 40 + rng.gauss(0, 5)
            write_event(
                layout.events_path,
                "selfplay",
                iter=i,
                games=50,
                avg_moves=round(avg_moves, 1),
                orange_win_rate=round(0.5 + rng.gauss(0, 0.05), 3),
            )

            # Fake training
            time.sleep(args.iter_seconds * 0.3)
            policy_loss = 3.0 * math.exp(-i / 20) + rng.gauss(0, 0.05)
            value_loss = 0.8 * math.exp(-i / 15) + rng.gauss(0, 0.02)
            write_event(
                layout.events_path,
                "train",
                iter=i,
                batches=400,
                policy_loss=round(policy_loss, 4),
                value_loss=round(value_loss, 4),
                total_loss=round(policy_loss + value_loss, 4),
            )

            # Fake eval (vs previous checkpoint)
            time.sleep(args.iter_seconds * 0.2)
            # Model gradually gets better but with noise.
            true_gain = 20 * (1 - math.exp(-i / 10))
            measured_gain = true_gain + rng.gauss(0, 10)
            fake_elo += measured_gain * 0.5
            wins = int(10 + measured_gain / 3)
            wins = max(0, min(20, wins))
            write_event(
                layout.events_path,
                "eval",
                iter=i,
                opponent="prev",
                wins=wins,
                losses=20 - wins,
                draws=0,
                score=wins / 20.0,
                elo=round(fake_elo, 1),
                elo_delta=round(measured_gain * 0.5, 1),
            )

            # Fake checkpoint
            ckpt_path = layout.checkpoints_dir / f"iter-{i:04d}.pt"
            ckpt_path.write_text(f"fake checkpoint for iter {i}\n")
            is_champion = wins >= 12
            write_event(
                layout.events_path,
                "checkpoint",
                iter=i,
                path=str(ckpt_path.relative_to(layout.root)),
                is_champion=is_champion,
            )

            # Fake sample game
            iter_samples = layout.samples_dir / f"iter-{i:04d}"
            iter_samples.mkdir(exist_ok=True)
            fake_game = iter_samples / "selfplay-game-01.json"
            fake_game.write_text(
                json.dumps({"iter": i, "moves": list(range(int(avg_moves)))}) + "\n"
            )
            write_event(
                layout.events_path,
                "sample_game",
                iter=i,
                path=str(fake_game.relative_to(layout.root)),
                kind="selfplay",
            )

            time.sleep(args.iter_seconds * 0.1)
            write_event(
                layout.events_path,
                "iter_end",
                iter=i,
                duration_s=round(time.monotonic() - iter_start, 2),
            )

        duration = time.monotonic() - start
        if _stop_requested:
            layout.write_status("stopped")
            write_event(
                layout.events_path,
                "run_end",
                status="stopped",
                duration_s=round(duration, 2),
            )
            return 0
        else:
            layout.write_status("finished")
            write_event(
                layout.events_path,
                "run_end",
                status="finished",
                duration_s=round(duration, 2),
            )
            return 0

    except Exception as e:  # noqa: BLE001
        layout.write_status("crashed")
        write_event(
            layout.events_path,
            "log",
            level="error",
            msg=f"trainer crashed: {type(e).__name__}: {e}",
        )
        write_event(
            layout.events_path,
            "run_end",
            status="crashed",
            duration_s=round(time.monotonic() - start, 2),
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
