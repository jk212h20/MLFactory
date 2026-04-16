/**
 * Boop TS-rules stdio bridge.
 *
 * Imports the authoritative BoopGame class from the Boop project and exposes it
 * over stdin/stdout JSON lines, so that the Python implementation in `rules.py`
 * can be validated against the canonical rules byte-for-byte.
 *
 * Protocol (one JSON object per line, both directions):
 *
 * Client -> bridge:
 *   { "op": "new_game" }
 *       -> { "ok": true, "state": <state> }
 *   { "op": "legal_actions", "game_id": N }
 *       -> { "ok": true, "actions": [...] }
 *   { "op": "step", "game_id": N, "action": <int> }
 *       -> { "ok": true, "state": <state> }  or  { "ok": false, "error": "..." }
 *   { "op": "state", "game_id": N }
 *       -> { "ok": true, "state": <state> }
 *   { "op": "ping" }
 *       -> { "ok": true }
 *
 * <state> is a plain JSON object mirroring the Python BoopState:
 *   {
 *     board: [int * 36],             // 0 empty, 1 O_kitten, 2 O_cat, 3 G_kitten, 4 G_cat
 *     orange_pool: [k, c, r],
 *     gray_pool: [k, c, r],
 *     to_play: 0 | 1,                // 0 orange, 1 gray
 *     phase: "playing" | "selecting_graduation" | "finished",
 *     winner: 0 | 1 | null,
 *     move_number: int,
 *     pending_options: [[int, int, int]]  // each inner = 3 linearised cell indices
 *   }
 *
 * Action encoding matches our Python rules.py:
 *   0..71  : place (kind * 36 + row * 6 + col)  where kind 0=kitten, 1=cat
 *   72+    : graduation-option index
 */

import { createInterface } from "node:readline";
// The Boop rules file targets CommonJS. Our bridge also uses CommonJS so the
// import is straightforward. tsx (via swc) compiles both on the fly.
import { BoopGame } from "../../../../../../Boop/server/src/game/GameState";

const BOARD_SIZE = 6;
const N_CELLS = 36;
const N_PLACE_ACTIONS = 72;

// ---------- piece encoding helpers ----------

type PieceColor = "orange" | "gray";
type PieceType = "kitten" | "cat";
type Piece = { color: PieceColor; type: PieceType };

function encodePiece(p: Piece | null): number {
    if (!p) return 0;
    if (p.color === "orange") return p.type === "kitten" ? 1 : 2;
    return p.type === "kitten" ? 3 : 4;
}

// ---------- game management ----------

// Synthetic socket IDs. We always drive moves by colour, not socket.
const ORANGE_SOCKET = "orange-bot";
const GRAY_SOCKET = "gray-bot";

class ManagedGame {
    game: BoopGame;
    constructor() {
        this.game = new BoopGame();
        this.game.addPlayer(ORANGE_SOCKET, "orange");
        this.game.addPlayer(GRAY_SOCKET, "gray");
        // After both added, phase should be "playing".
    }

    private socketFor(color: PieceColor): string {
        return color === "orange" ? ORANGE_SOCKET : GRAY_SOCKET;
    }

    snapshot() {
        const raw = this.game.getState();
        const board: number[] = new Array(N_CELLS).fill(0);
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                board[r * BOARD_SIZE + c] = encodePiece(raw.board[r][c]);
            }
        }
        const orangePool: [number, number, number] = raw.players.orange
            ? [
                raw.players.orange.kittensInPool,
                raw.players.orange.catsInPool,
                raw.players.orange.kittensRetired,
              ]
            : [0, 0, 0];
        const grayPool: [number, number, number] = raw.players.gray
            ? [
                raw.players.gray.kittensInPool,
                raw.players.gray.catsInPool,
                raw.players.gray.kittensRetired,
              ]
            : [0, 0, 0];

        const to_play = raw.currentTurn === "orange" ? 0 : 1;
        const winner =
            raw.winner === null ? null : raw.winner === "orange" ? 0 : 1;

        let pendingOptions: number[][] = [];
        if (raw.phase === "selecting_graduation" && raw.pendingGraduationOptions) {
            pendingOptions = raw.pendingGraduationOptions.map((opt) =>
                opt.map((cell) => cell.row * BOARD_SIZE + cell.col)
            );
        }

        return {
            board,
            orange_pool: orangePool,
            gray_pool: grayPool,
            to_play,
            phase: raw.phase, // matches "playing" | "selecting_graduation" | "finished"
            winner,
            move_number: this.moveNumber,
            pending_options: pendingOptions,
        };
    }

    // Track move number ourselves — the TS game doesn't expose it.
    moveNumber = 0;

    legalActions(): number[] {
        const s = this.game.getState();
        if (s.phase === "finished") return [];
        if (s.phase === "selecting_graduation") {
            const n = s.pendingGraduationOptions?.length ?? 0;
            const out: number[] = [];
            for (let i = 0; i < n; i++) out.push(N_PLACE_ACTIONS + i);
            return out;
        }
        // phase === playing
        const player =
            s.currentTurn === "orange" ? s.players.orange : s.players.gray;
        if (!player) return [];
        const kittens = player.kittensInPool;
        const cats = player.catsInPool;
        const out: number[] = [];
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                if (s.board[r][c] !== null) continue;
                const cellIdx = r * BOARD_SIZE + c;
                if (kittens > 0) out.push(0 * N_CELLS + cellIdx);
                if (cats > 0) out.push(1 * N_CELLS + cellIdx);
            }
        }
        return out;
    }

    step(action: number): { ok: true } | { ok: false; error: string } {
        const s = this.game.getState();
        if (s.phase === "finished") return { ok: false, error: "terminal" };

        if (s.phase === "selecting_graduation") {
            const idx = action - N_PLACE_ACTIONS;
            const player =
                s.pendingGraduationPlayer === "orange"
                    ? s.players.orange
                    : s.players.gray;
            if (!player) return { ok: false, error: "no player" };
            const res = this.game.selectGraduation(player.socketId, idx);
            if (!res.valid) {
                return { ok: false, error: res.error ?? "invalid" };
            }
            this.moveNumber += 1;
            return { ok: true };
        }

        // phase playing
        if (action < 0 || action >= N_PLACE_ACTIONS) {
            return { ok: false, error: "non-place action in playing phase" };
        }
        const kind = Math.floor(action / N_CELLS);
        const cellIdx = action % N_CELLS;
        const row = Math.floor(cellIdx / BOARD_SIZE);
        const col = cellIdx % BOARD_SIZE;
        const pieceType: PieceType = kind === 0 ? "kitten" : "cat";
        const player =
            s.currentTurn === "orange" ? s.players.orange : s.players.gray;
        if (!player) return { ok: false, error: "no player" };
        const res = this.game.placePiece(player.socketId, row, col, pieceType);
        if (!res.valid) {
            return { ok: false, error: res.error ?? "invalid" };
        }
        this.moveNumber += 1;
        return { ok: true };
    }
}

// ---------- stdio protocol loop ----------

const games = new Map<number, ManagedGame>();
let nextGameId = 0;

function handleLine(line: string): string {
    let req: Record<string, unknown>;
    try {
        req = JSON.parse(line);
    } catch (e) {
        return JSON.stringify({ ok: false, error: "bad json" });
    }
    const op = req.op as string;
    try {
        if (op === "ping") return JSON.stringify({ ok: true });
        if (op === "new_game") {
            const game = new ManagedGame();
            const id = nextGameId++;
            games.set(id, game);
            return JSON.stringify({ ok: true, game_id: id, state: game.snapshot() });
        }
        const id = req.game_id as number;
        const game = games.get(id);
        if (!game) return JSON.stringify({ ok: false, error: `unknown game_id ${id}` });
        if (op === "legal_actions") {
            return JSON.stringify({ ok: true, actions: game.legalActions() });
        }
        if (op === "state") {
            return JSON.stringify({ ok: true, state: game.snapshot() });
        }
        if (op === "step") {
            const action = req.action as number;
            const res = game.step(action);
            if (!res.ok) return JSON.stringify({ ok: false, error: res.error });
            return JSON.stringify({ ok: true, state: game.snapshot() });
        }
        if (op === "close") {
            games.delete(id);
            return JSON.stringify({ ok: true });
        }
        return JSON.stringify({ ok: false, error: `unknown op ${op}` });
    } catch (e) {
        return JSON.stringify({ ok: false, error: `exception: ${(e as Error).message}` });
    }
}

const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
});
rl.on("line", (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    const response = handleLine(trimmed);
    process.stdout.write(response + "\n");
});
rl.on("close", () => {
    process.exit(0);
});
