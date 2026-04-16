---
type: technique
status: stable
created: 2026-04-16
updated: 2026-04-16
tags: [mcts, uct, search]
links:
  sources: [browne2012-mcts-survey]
  answers: []
  used_in: [src/mlfactory/agents/mcts.py]
  validated_by: [results/phase1-mcts-baseline.md, insights/2026-04-16-mcts-logarithmic-in-sims]
---

# Vanilla UCT MCTS

## What it does
Best-first game-tree search that iteratively builds an asymmetric tree by running many simulations ("playouts"), using the UCT bandit formula to trade off exploitation of good branches against exploration of under-sampled ones. No evaluation function needed — uses random rollouts.

## When to use
- **Early baseline** for any game before bringing out neural nets. If vanilla UCT can't beat a random agent decisively, something is wrong with the game interface, not the search.
- Games with moderate branching factor (~10–50) and short enough games that random rollouts finish in reasonable time.
- When you need a strong opponent but have no training data or compute for learning.

## When NOT to use
- Games with very long rollouts (random playouts give noise, not signal) → needs a learned value function; go straight to AlphaZero-style PUCT.
- Games with sharp tactics / trap states → UCT misses narrow refutations at low budgets. Use neural-guided search.
- When you have a good evaluation function already — minimax with α-β pruning may be stronger at the same compute.

## Pattern / pseudocode

```python
def mcts(root_state, n_simulations, c=1.4):
    root = Node(root_state)
    for _ in range(n_simulations):
        # 1. SELECT: descend tree via UCT until unexpanded / terminal
        node = root
        path = [node]
        while node.children and not node.state.is_terminal():
            node = max(node.children,
                       key=lambda ch: uct_score(ch, node.visits, c))
            path.append(node)

        # 2. EXPAND: add one child if not terminal
        if not node.state.is_terminal():
            action = node.untried_actions.pop()
            child = Node(node.state.step(action), parent=node, action=action)
            node.children.append(child)
            path.append(child)
            node = child

        # 3. SIMULATE: random playout to terminal
        reward = random_rollout(node.state)

        # 4. BACKPROP: update stats along path (flip sign for opponent)
        for n in reversed(path):
            n.visits += 1
            n.value_sum += reward if n.state.to_play == winning_player else -reward

    return max(root.children, key=lambda ch: ch.visits).action   # "robust child"


def uct_score(child, parent_visits, c):
    if child.visits == 0:
        return float("inf")
    exploit = child.value_sum / child.visits
    explore = c * sqrt(log(parent_visits) / child.visits)
    return exploit + explore
```

## Common pitfalls
- **Sign of reward in backprop**: In a 2-player zero-sum game, flip the sign every ply or track side-to-move explicitly. Off-by-one here silently destroys strength.
- **Exploration constant `c`**: Theoretical `√2 ≈ 1.414` is for rewards in [0, 1]. If you use `[-1, +1]`, rescale or retune. Typical useful range: 0.5–2.0. Sweep it against a fixed opponent.
- **Robust child vs max child**: Pick the final move by highest visit count (robust), not highest mean value (max). Visit count is the more stable estimator at modest budgets.
- **Trap states**: UCT can miss narrow tactical refutations that require deep narrow lines. If your game has these, raise simulation count drastically or switch to neural-guided search.
- **Rollout policy matters**: Uniform-random rollouts are an unbiased but noisy estimator. Light domain heuristics in rollouts can help — but **stronger rollouts can hurt** if they reduce diversity without being exactly correct.

## Implementation in MLFactory
- `src/mlfactory/agents/mcts.py` — `MCTSAgent`. Vanilla UCT with uniform-random rollouts.
- Value convention: `node.value_sum / node.visits` is from the perspective of the player
  who **moved into** that node. This makes UCT selection from a parent a plain maximisation
  (the parent is choosing the child with the best score for whoever just made the move).
- Backpropagation flips sign every ply as it walks up from the rollout terminal.
- Validated on Connect 4: monotonic ELO ladder across {50, 200, 800} simulation budgets,
  ~300 ELO per 4× budget (see [[insights/2026-04-16-mcts-logarithmic-in-sims]]).

## Sources
- [[sources/browne2012-mcts-survey]]

## See also
- [[techniques/puct-with-priors]] — the neural-prior upgrade (AlphaZero/ExIt). _(to be written when we implement Phase 3.)_
