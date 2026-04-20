<h1 align="center"><strong>promptloop</strong></h1>

<p align="center">
  <em>Autonomous prompt optimization — an AI agent iterates on your system prompt overnight so you don't have to.</em>
</p>

---

## Overview

**promptloop** is the prompt-engineering counterpart to [autoresearch](https://github.com/karpathy/autoresearch). Instead of training a model, you give an AI agent a fixed evaluation harness and let it iterate on a single `system_prompt.txt` file autonomously overnight.

The idea is simple: the agent modifies the prompt, runs it against a fixed dataset, checks if the F1 score improved, keeps or discards the change, and repeats. You wake up to a log of experiments and — hopefully — a significantly better prompt.

The metric is **token-level F1** on a fixed 100-example subset of [SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2) — vocab-size-independent, fully deterministic (`temperature=0.0`), and automatically computed. No human judges. No ambiguity.

---

## How It Works

The repo has four files that matter:

| File | Role | Who touches it |
|------|------|----------------|
| `prepare.py` | Download SQuAD v2, `compute_f1`, fixed dataset loader | No one — frozen harness |
| `evaluate.py` | Call LLM on 100 examples, compute average F1, print summary | No one — run as-is |
| `system_prompt.txt` | The system prompt being optimized | Agent only |
| `program.md` | Agent instructions — your "research org policy" | You |

**The loop:**

```
agent reads program.md
  └─ modifies system_prompt.txt
      └─ git commit
          └─ uv run evaluate.py > run.log 2>&1  (~2 min)
              └─ grep "^f1_score:" run.log
                  ├─ improved → KEEP
                  └─ same/worse → git reset, try something else
```

Results accumulate in `results.tsv`. Analysis and visualization in `analysis.ipynb`.

---

## Repository Structure

```
promptloop/
├── prepare.py        # Fixed: SQuAD v2 download, compute_f1, dataset loader
├── evaluate.py       # Fixed: LLM evaluation loop, F1 aggregation, summary
├── system_prompt.txt # Variable: the only file the agent edits
├── config.py         # Provider, model, EVAL_SAMPLES, temperature
├── program.md        # Agent instructions (edit this to improve your "research org")
├── analysis.ipynb    # Visualize results.tsv — F1 trajectory + stats
├── pyproject.toml    # Dependencies
└── results.tsv       # Experiment log (git-untracked, stays local)
```

---

## Quick Start

**Requirements:** Python 3.10+, [`uv`](https://astral.sh/uv), an OpenAI or Anthropic API key.

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download SQuAD v2 evaluation data (one-time, ~1 min)
uv run prepare.py

# 4. Set your API key
export OPENAI_API_KEY="sk-..."     # or ANTHROPIC_API_KEY

# 5. Run a single evaluation manually to verify setup (~2 min)
uv run evaluate.py
```

You should see per-example F1 scores and a final summary:

```
[promptloop] Provider: openai | Model: gpt-4.1
[promptloop] Starting evaluation (100 examples)...

  example 001/100 | f1: 0.82 | pred: "in the late 1990s" | truth: "in the late 1990s"
  example 002/100 | f1: 0.00 | pred: "unanswerable" | truth: "Houston"
  ...

---
f1_score:     0.730755
total_tokens: 24302
eval_samples: 100
model:        gpt-4.1
```

---

## Running the Agent

Configure your model and provider in `config.py`:

```python
PROVIDER = "openai"           # "openai" | "anthropic"
MODEL = "gpt-4.1"
API_KEY_ENV = "OPENAI_API_KEY"
EVAL_SAMPLES = 100            # examples per evaluation run
TEMPERATURE = 0.0             # keep this — ensures reproducibility
```

Then spin up Claude Code (or any agentic coding assistant) in this repo and say:

```
Have a look at program.md and let's kick off a new experiment!
```

The agent will:

1. Create a dated branch (`promptloop/apr20`)
2. Establish a baseline by running `evaluate.py` unchanged
3. Loop autonomously — modifying `system_prompt.txt`, committing, evaluating, keeping or discarding

You can interrupt at any time. Results so far are always in `results.tsv`.

---

## Viewing Results

After experiments complete, open `analysis.ipynb`:

```bash
jupyter notebook analysis.ipynb
```

You'll get:

- F1 score trajectory (green = keep, red = discard, gray = crash)
- Keep / discard / crash counts
- Best experiment commit and description
- Total improvement over baseline
- Token cost breakdown

---

## Design Choices

**Single file to modify.** The agent only touches `system_prompt.txt`. Diffs are one-liners. Rollback is `git reset`. Nothing surprising happens.

**Fixed evaluation set.** `seed=42` selects the same 100 examples every run. With `temperature=0.0`, any F1 improvement — however small — is a real signal, not noise.

**Determinism over coverage.** 100 examples is small but perfectly reproducible. Running 50 experiments at 100 examples beats running 5 experiments at 1000 examples for the same API cost.

**Simplicity criterion.** A simpler prompt that gains +0.001 F1 beats a complex prompt that gains +0.005 F1. Fewer tokens, easier to understand, easier to iterate on. The agent is instructed to prefer removal over addition.

**Provider-agnostic.** Switch between OpenAI and Anthropic by changing two lines in `config.py`. The evaluation harness is identical either way.

**`program.md` is yours to evolve.** The default is intentionally minimal — a baseline research org policy. The interesting question is: what instructions make the agent most effective? That's the meta-experiment.

---

## Cost Estimate

Each evaluation run costs approximately:

| | Per run (100 examples) |
|--|--|
| Input tokens | ~33,000 |
| Output tokens | ~1,500 |
| Cost (gpt-4.1) | ~$0.08 |

| Scenario | Experiments | Cost |
|--|--|--|
| Quick test | 5 | ~$0.40 |
| Afternoon session | 20 | ~$1.60 |
| Overnight run | 50–80 | ~$4–6 |
