# MLFactory Wiki

> **North Star**: MLFactory is a self-improving automated games strategy research factory. **The wiki is where our learning compounds.**

This is a Karpathy-style living research knowledge base — not a static doc pile. It is driven by research **questions** we encounter as we work, grounded in **sources** we actually read, and distilled into reusable **techniques**. **Trails** narrate the journey. **Insights** capture surprising findings from experiments.

When we work on a problem, we write. When we write, we link. When we link, future-us (or an agent) can retrace the reasoning without redoing the work.

---

## Directory structure

```
wiki/
├── README.md        # this file — the workflow
├── INDEX.md         # cross-referenced table of contents
├── sources/         # distilled external material (papers, blogs, talks)
├── questions/       # Q-NNN numbered research questions, each with an answer + sources
├── techniques/      # how-to pages, promoted after a technique is used twice
├── advice/          # meta-advice (debugging RL, getting started, pitfalls)
├── games/<game>/    # per-game discoveries, strategy notes, known positions
├── trails/          # narrative logs: "we wanted X, read Y, found Z"
└── insights/        # dated surprising findings from experiments, each with repro recipe
```

---

## Frontmatter contract

Every markdown file in `wiki/` starts with YAML frontmatter:

```yaml
---
type: source | question | technique | advice | game-note | trail | insight
status: seed | draft | stable
created: 2026-04-16
updated: 2026-04-16
tags: [mcts, self-play]
links:
  sources: [silver2017-alphazero]
  answers: [Q-001]            # questions this page answers
  uses: [mcts-uct]            # techniques this page uses
provenance:                   # sources only
  url: https://...
  accessed: 2026-04-16
  read_level: abstract | skim | full
speculation: false            # true if claims go beyond what's in the sources
---
```

The `speculation` flag is important: it's fine to speculate, but it must be labelled. No invented numbers or citations.

---

## The five content types

### `sources/` — what we've read

One file per paper, blog post, talk, book chapter. Each contains:
- The one-sentence claim of the work.
- 3–7 key takeaways.
- Direct quotes with locations (or an explicit note that we only read an abstract).
- "Numbers worth remembering" — cited or flagged.
- Our open questions it left us with.
- How this maps onto MLFactory (Boop, our agents, our roadmap).

Provenance is mandatory: URL, access date, and the honest `read_level` (did we read the full PDF, skim it, or only read an abstract?). Claims that rely on specific numbers that we didn't verify in the source get explicitly flagged.

### `questions/` — what we needed to know

Named `Q-NNN-slug.md`, numbered sequentially. Every time we face a non-trivial design decision or empirical question, we open one.

Each contains:
- The question in one sentence.
- The context that raised it.
- Sources consulted (links into `sources/`).
- The answer, with honest confidence.
- Follow-up questions.
- Which techniques or code decisions fell out of this.

**Rule**: do not answer a significant research question without producing a `Q-NNN` page. This is how our knowledge compounds.

### `techniques/` — how we do things

A technique page is a distilled how-to. **Promote to a technique after a pattern is used twice.** Before that, the knowledge lives in the question that introduced it.

Each contains:
- What the technique does in one sentence.
- Pseudocode or pattern.
- When to use it, when not to.
- Common pitfalls.
- Sources that introduced or validate it.
- Where we use it in `src/mlfactory/`.

### `insights/` — what surprised us

An insight is a finding from an experiment that changed our understanding. Not every experiment produces one; most confirm what we already expected. The ones that don't are valuable.

Each contains:
- The surprise in one sentence.
- What we expected vs. what we found.
- The experiment that produced it (link into `experiments/<game>/<run-id>/` and the report).
- Repro recipe (config hash, git SHA, seed).
- Implications.

Insights feed `INSIGHTS.md` as a dated chronological index.

### `trails/` — the journey

A trail is a narrative log, roughly monthly. It reconstructs the order in which questions arose and were answered — something a linear outline can't capture. When future-us looks back and wonders "why did we decide X?", trails are where we go.

### `advice/` — the cross-cutting wisdom

Meta-level pages about how to do research well in this setting: debugging RL, getting started on a new game, reading papers efficiently, compute budgets, common pitfalls. These are written by us, for us, grounded in experience.

### `games/<game>/` — per-game knowledge

Everything we've discovered about a specific game: rules summary, symmetries, state-space estimate, notable positions, opening theory we derived, endgame tablebases, and claims about solutions or advantages. Each file cites the experiments it rests on.

---

## Workflow in practice

When a research question hits you:

1. **Check `questions/` first.** Has this already been answered?
2. **Check `sources/` and `techniques/`** for adjacent ground.
3. If new: **open `Q-NNN-slug.md` with status `seed`**.
4. **Ingest any new sources** into `sources/` with honest provenance.
5. **Write the answer**, linking those sources; promote status to `draft` → `stable`.
6. **If the pattern emerges twice**, promote to `techniques/`.
7. **If the answer reveals a surprise**, write an `insights/` entry.
8. **Append a one-paragraph entry to the current month's trail.**
9. **Update `INDEX.md`** (or let the regen script do it).

---

## Naming conventions

- Sources: `author-year-slug.md` — e.g., `silver2017-alphazero.md`.
- Questions: `Q-NNN-slug.md` — zero-padded to 3 digits.
- Techniques: `slug.md` — e.g., `puct-with-priors.md`.
- Advice: `slug.md` — e.g., `debugging-rl.md`.
- Games: under `games/<game-slug>/`, use kebab-case.
- Trails: `YYYY-MM-slug.md`.
- Insights: `YYYY-MM-DD-slug.md`.

---

## Not in the wiki

- Full paper PDFs (link to them, don't store).
- Experiment raw data (lives in `experiments/`, gitignored).
- Code (lives in `src/`, linked to from wiki pages).
- Opinions without evidence (if you have to, mark `speculation: true`).

---

## Meta-goal

In six months, when we want to train an agent for a new game, the wiki should let us stand on all prior learning in a single afternoon. That's the test.
