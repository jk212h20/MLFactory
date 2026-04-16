"""Python client for the TypeScript Boop rules bridge.

Spawns `npx tsx bridge.ts` as a subprocess and talks to it via JSON lines over
stdin/stdout. One long-lived subprocess can host many concurrent games, each
referenced by its integer game_id.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

BRIDGE_DIR = Path(__file__).parent / "bridge"


class BridgeError(RuntimeError):
    """Raised when the TS bridge returns an error."""


class BoopBridge:
    """Long-lived Node subprocess that runs the authoritative TS rules."""

    def __init__(self, bridge_dir: Path | None = None, node_bin: str = "npx") -> None:
        self.bridge_dir = bridge_dir or BRIDGE_DIR
        if not (self.bridge_dir / "node_modules").exists():
            raise BridgeError(
                f"bridge dependencies not installed. Run: cd {self.bridge_dir} && npm install"
            )
        # We call `node_modules/.bin/tsx` directly to avoid npx's startup overhead.
        tsx = self.bridge_dir / "node_modules" / ".bin" / "tsx"
        cmd = [str(tsx), "bridge.ts"] if tsx.exists() else [node_bin, "tsx", "bridge.ts"]
        env = os.environ.copy()
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.bridge_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        # Verify it's alive.
        reply = self._call({"op": "ping"})
        if not reply.get("ok"):
            raise BridgeError(f"bridge ping failed: {reply}")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self) -> BoopBridge:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    # -- low-level ----------------------------------------------------

    def _call(self, payload: dict) -> dict:
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        msg = json.dumps(payload) + "\n"
        try:
            self.proc.stdin.write(msg)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise BridgeError(f"bridge stdin write failed: {e}") from e
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise BridgeError(f"bridge died. stderr:\n{stderr}")
        return json.loads(line)

    # -- high-level ---------------------------------------------------

    def new_game(self) -> tuple[int, dict]:
        """Start a new game. Returns (game_id, initial_state_dict)."""
        r = self._call({"op": "new_game"})
        if not r.get("ok"):
            raise BridgeError(f"new_game failed: {r}")
        return r["game_id"], r["state"]

    def legal_actions(self, game_id: int) -> list[int]:
        r = self._call({"op": "legal_actions", "game_id": game_id})
        if not r.get("ok"):
            raise BridgeError(f"legal_actions failed: {r}")
        return r["actions"]

    def step(self, game_id: int, action: int) -> dict:
        """Apply action; returns the new state dict."""
        r = self._call({"op": "step", "game_id": game_id, "action": action})
        if not r.get("ok"):
            raise BridgeError(f"step({action}) failed: {r.get('error')}")
        return r["state"]

    def state(self, game_id: int) -> dict:
        r = self._call({"op": "state", "game_id": game_id})
        if not r.get("ok"):
            raise BridgeError(f"state failed: {r}")
        return r["state"]

    def close_game(self, game_id: int) -> None:
        self._call({"op": "close", "game_id": game_id})
