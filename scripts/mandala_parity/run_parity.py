"""Run N Mandala games on both JS and Python engines with identical RNG
seeds and random action choices, compare step-by-step. Exits 0 on full
parity, 1 on any divergence.

Usage:
    uv run python scripts/mandala_parity/run_parity.py --n-games 100 [--max-turns 500]

This is the Mandala analogue of the 10k-game Boop TS↔Python parity harness
that caught a bunch of subtle rule bugs in phase 2. Same principle: pick
random action indices in Python, replay both engines, compare after every
step.

Reports:
  - games compared
  - total turns compared
  - first divergence (seed, turn index, diff summary) — if any
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

# Paths
MANDALA_DIR = Path(__file__).resolve().parent
JS_RUNNER = MANDALA_DIR / "parity_runner.mjs"
PY_RUNNER = MANDALA_DIR / "parity_runner.py"
PROJECT_ROOT = MANDALA_DIR.parent.parent  # MLFactory/


def run_js(seed: int, choices: list[int]) -> dict:
    """Invoke the Node parity runner and parse its JSON output."""
    inp = json.dumps({"seed": seed, "actionChoices": choices})
    res = subprocess.run(
        ["node", str(JS_RUNNER)],
        input=inp,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=60,
    )
    if res.returncode != 0:
        raise RuntimeError(f"JS runner failed: {res.stderr}")
    return json.loads(res.stdout)


def run_py(seed: int, choices: list[int]) -> dict:
    """Invoke the Python parity runner as a subprocess so it mirrors JS exactly."""
    inp = json.dumps({"seed": seed, "actionChoices": choices})
    res = subprocess.run(
        [sys.executable, str(PY_RUNNER)],
        input=inp,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=60,
    )
    if res.returncode != 0:
        raise RuntimeError(f"PY runner failed: {res.stderr}")
    return json.loads(res.stdout)


def diff_states(js: dict, py: dict) -> str | None:
    """Return a short description of the first divergence found between two
    state dicts, or None if they are byte-equal.

    Deep-equality via JSON roundtrip (both sides produced JSON, so this is
    essentially string comparison). On inequality, probe a few likely
    divergence points for a readable summary."""
    # Fast path: JSON serialise both and compare.
    js_s = json.dumps(js, sort_keys=True)
    py_s = json.dumps(py, sort_keys=True)
    if js_s == py_s:
        return None

    # Find the diff by inspecting likely divergence keys.
    def probe(path, a, b) -> list[str]:
        out = []
        if a == b:
            return out
        if type(a) != type(b):
            out.append(f"{path}: type mismatch {type(a).__name__} vs {type(b).__name__}")
            return out
        if isinstance(a, dict):
            keys = sorted(set(a) | set(b))
            for k in keys:
                if a.get(k) != b.get(k):
                    out.extend(probe(f"{path}.{k}", a.get(k), b.get(k)))
        elif isinstance(a, list):
            if len(a) != len(b):
                out.append(f"{path}: list len {len(a)} vs {len(b)}")
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        out.extend(probe(f"{path}[{i}]", x, y))
                        if len(out) > 6:
                            break
        else:
            out.append(f"{path}: {a!r} vs {b!r}")
        return out

    diffs = probe("", js, py)
    return "\n    ".join(diffs[:12]) or "(empty probe; full json differs)"


def compare_one(seed: int, max_turns: int) -> tuple[bool, str, int]:
    """Run both engines on the same seed. Returns (ok, description, turns_compared)."""
    rng = random.Random(seed)
    choices = [rng.randint(0, 10**9) for _ in range(max_turns)]
    js_result = run_js(seed, choices)
    py_result = run_py(seed, choices)

    js_steps = js_result["steps"]
    py_steps = py_result["steps"]

    # Number of steps should match.
    if len(js_steps) != len(py_steps):
        return (
            False,
            f"step count mismatch: js={len(js_steps)}, py={len(py_steps)}",
            min(len(js_steps), len(py_steps)),
        )

    for i, (js_step, py_step) in enumerate(zip(js_steps, py_steps)):
        # Compare the state object (ignore the `action` key if both match).
        d = diff_states(js_step.get("state"), py_step.get("state"))
        if d is not None:
            return (
                False,
                f"state diverges at step {i}:\n    {d}",
                i,
            )

    # Also check winner.
    if js_result.get("winner") != py_result.get("winner"):
        return (
            False,
            f"winner disagrees: js={js_result.get('winner')} py={py_result.get('winner')}",
            len(js_steps),
        )

    return True, "ok", len(js_steps)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--first-seed", type=int, default=0)
    args = parser.parse_args()

    total_turns = 0
    for i in range(args.n_games):
        seed = args.first_seed + i
        try:
            ok, msg, turns = compare_one(seed, args.max_turns)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] seed={seed} subprocess error: {e}", file=sys.stderr)
            return 1
        total_turns += turns
        if not ok:
            print(f"[FAIL] seed={seed} turn_failed≈{turns}: {msg}", file=sys.stderr)
            return 1
        if (i + 1) % max(1, args.n_games // 20) == 0:
            print(f"  {i + 1}/{args.n_games} games ok  ({total_turns} total turns)")

    print(f"\nPARITY OK: {args.n_games} games, {total_turns} turns, zero divergences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
