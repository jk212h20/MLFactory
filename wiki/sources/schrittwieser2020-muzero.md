---
type: source
status: stable
created: 2026-04-16
updated: 2026-04-16
tags: [muzero, model-based-rl, planning, mcts]
provenance:
  url: https://arxiv.org/abs/1911.08265
  accessed: 2026-04-16
  read_level: skim
speculation: false
notes: |
  Nature page paywalled past the abstract; arXiv abstract read. Hyperparameter numbers
  (K=5 unroll, 800 sims board / 50 sims Atari) are my recollection from prior reading and
  are flagged for verification against Methods / pseudocode.py before citing.
---

# Schrittwieser et al. 2020 — MuZero

## One-sentence claim
MuZero combines MCTS with a *learned* latent-space dynamics model that predicts only planning-relevant quantities (policy, value, reward), achieving AlphaZero-level play on Go/Chess/Shogi without knowing the rules, and SOTA on Atari-57.

## Bibliographic
- Authors: Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, David Silver (DeepMind)
- Venue/year: Nature 588, 604–609 (23 Dec 2020); arXiv:1911.08265 (Nov 2019, v2 Feb 2020)
- Link: https://arxiv.org/abs/1911.08265 ; https://doi.org/10.1038/s41586-020-03051-4
- Pseudocode: ancillary `pseudocode.py` on arXiv v2 and Nature SI.

## Core advance beyond AlphaZero
AlphaZero requires a perfect simulator at plan time (the rules). MuZero replaces that with a model learned *end-to-end to be useful for planning*, operating in an abstract latent state space — not a reconstructed observation space. Three learned networks:

- **Representation `h`**: observation history → latent state `s⁰`.
- **Dynamics `g`**: `(sᵏ, aᵏ⁺¹)` → `(sᵏ⁺¹, rᵏ⁺¹)`.
- **Prediction `f`**: `sᵏ` → `(πᵏ, vᵏ)`.

Key design choice: the latent state has **no semantic grounding** — there is no reconstruction loss tying it to observations. It only has to produce correct policy/value/reward predictions under rollout. That's what lets it work on visually complex Atari where pixel-level world models had underperformed model-free.

## Training targets
- **Policy**: MCTS visit-count distribution at the root (as in AlphaZero).
- **Value**: n-step bootstrapped return (TD for Atari; final game outcome for board games).
- **Reward**: observed reward (zero for board games except terminal).
- **Loss**: sum over K unrolled steps of policy + value + reward losses + L2 reg.

## Direct quotes (from abstracts — full paper not read)
> "Constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence."

> "MuZero learns a model that, when applied iteratively, predicts the quantities most directly relevant to planning: the reward, the action-selection policy, and the value function." (arXiv abstract)

> "When evaluated on Go, chess and shogi, without any knowledge of the game rules, MuZero matched the superhuman performance of the AlphaZero algorithm that was supplied with the game rules." (arXiv abstract)

## Numbers worth remembering
- **57** Atari games (ALE benchmark); SOTA at publication.
- **K = 5** unrolling steps at training (standard config; ⚠️ not verified from abstract).
- Board games: matches AlphaZero ELO in Go, Chess, Shogi.
- MCTS sims/move: 800 for board games, 50 for Atari (standard config; ⚠️ not verified from abstract).

## Our open questions
- Sensitivity to K (unroll depth) at train time vs sim count at inference.
- Latent-space drift: do old replay trajectories go stale under changing h, g, f?
- Two-player zero-sum handling (value negation across plies); what does MuZero do at stochastic or imperfect-info decision nodes? (Vanilla MuZero does not — see Stochastic MuZero 2022.)
- Relationship to Predictron (Silver 2017) and Value Prediction Networks (Oh 2017) — novelty is scale + MCTS-in-latent-space, not value-equivalent model per se.

## How this maps to MLFactory
- For **Boop**: we HAVE a perfect simulator (rules simple, deterministic, perfect-info, zero-sum). **AlphaZero is strictly the right starting point** — no reason to pay MuZero's tax of learning dynamics when `step(state, action)` is free and exact.
- **When would MuZero matter for MLFactory later?**
  - Imperfect-info games (TheDuke with hidden stacks, IntegerPoker): but vanilla MuZero is for perfect-info — you'd want Player-of-Games / ReBeL / Stochastic MuZero. MuZero alone is not the right tool here.
  - Stochastic transitions: needs Stochastic MuZero (Antonoglou 2022).
  - Games where rules are expensive to simulate: learned dynamics in latent space could amortize. Not our situation.
- **Real takeaway**: the `representation / dynamics / prediction` factoring is a clean architectural pattern even if we don't learn the dynamics — our AlphaZero net is effectively `h ∘ f`, with `g` replaced by the true simulator. Worth keeping that decomposition explicit in code so a future MuZero-style swap-in is local.
