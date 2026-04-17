// Parity-harness runner for Mandala.
//
// Reads JSON from stdin: {seed: int, actionChoices: [int, ...]}.
// Uses a mulberry32 PRNG seeded with `seed` to replace Math.random so
// the JS engine's shuffles match the Python side's mulberry32 bit-for-bit.
// Plays a game, taking the action at index actionChoices[t] from the
// legal-action list at turn t. Writes a JSON stream of per-step states
// (and final winner) to stdout.
//
// Invoke:  node parity_runner.js  (reads stdin, writes stdout).

import { readFileSync } from 'node:fs';
import {
  createGame,
  getValidActions,
  performAction,
  getWinner,
} from '../../../mandala-web/game.js';

// --- Patch Math.random to a seeded mulberry32 that matches the Python impl.
function installDeterministicRandom(seed) {
  let state = seed >>> 0;
  Math.random = function () {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= (t + Math.imul(t ^ (t >>> 7), t | 61)) >>> 0;
    return (((t ^ (t >>> 14)) >>> 0)) / 4294967296;
  };
}

function actionObjectsFromValid(valid) {
  const list = [];
  for (const a of valid.buildMountain) list.push({ type: 'build_mountain', ...a });
  for (const a of valid.growField) list.push({ type: 'grow_field', ...a });
  for (const a of valid.discardRedraw) list.push({ type: 'discard_redraw', ...a });
  for (const color of valid.claimColor) list.push({ type: 'claim_color', color });
  return list;
}

const input = JSON.parse(readFileSync(0, 'utf-8'));
const seed = input.seed >>> 0;
const choices = input.actionChoices;

installDeterministicRandom(seed);

let state = createGame('p0', 'p1');

const steps = [];
steps.push({ turn: 0, state });

for (let i = 0; i < choices.length; i++) {
  if (state.phase === 'ended') break;
  const valid = getValidActions(state);
  const flat = actionObjectsFromValid(valid);
  if (flat.length === 0) break;
  const idx = choices[i] % flat.length;
  const action = flat[idx];
  const res = performAction(state, action);
  if (!res.success) {
    steps.push({ turn: i + 1, error: res.error, state });
    break;
  }
  state = res.newState;
  steps.push({ turn: i + 1, action, state });
}

const winner = getWinner(state);
process.stdout.write(JSON.stringify({ steps, winner }));
