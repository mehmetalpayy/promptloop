# promptloop — Design Spec

**Date:** 2026-04-14  
**Status:** Approved

---

## Overview

`promptloop` is an autonomous prompt optimization system that mirrors the autoresearch repo's run → measure → keep/discard loop. Instead of modifying a training script, the agent modifies a single `system_prompt.txt` file and evaluates its quality on a fixed SQuAD v2 subset using token-level F1 score.

---

## Goals

- Autonomously iterate on a system prompt overnight without human intervention
- Measure improvement with a single, reproducible numerical metric (F1)
- Log all experiments to `results.tsv` for analysis
- Visualize experiment history via `analysis.ipynb`
- Support multiple LLM providers via config

---

## File Structure

```
promptloop/
├── prepare.py        # FIXED: download SQuAD v2, compute_f1, dataset loader
├── evaluate.py       # RUNS: call LLM, compute F1 over N examples, print summary
├── system_prompt.txt # AGENT MODIFIES: the only file the agent changes
├── config.py         # provider, model, EVAL_SAMPLES, MAX_TOKENS, TEMPERATURE
├── program.md        # agent instructions (mirrors autoresearch/program.md)
├── results.tsv       # untracked by git, experiment log
├── run.log           # last evaluate.py stdout
├── analysis.ipynb    # visualization notebook
└── pyproject.toml    # dependencies
```

---

## Autoresearch Parallel

| autoresearch    | promptloop          |
|-----------------|---------------------|
| `prepare.py`    | `prepare.py`        |
| `train.py`      | `evaluate.py`       |
| hyperparameters | `system_prompt.txt` |
| `val_bpb` (↓)  | `f1_score` (↑)      |
| `program.md`    | `program.md`        |

---

## Component Details

### prepare.py (fixed, do not modify)

- Downloads SQuAD v2 validation split from HuggingFace datasets
- Saves to `~/.cache/promptloop/squad_v2.jsonl`
- Selects `EVAL_SAMPLES` examples with `seed=42` — same examples every run
- Exposes `compute_f1(prediction, ground_truth)` — official SQuAD token-level F1
- Exposes `load_eval_dataset()` — returns the fixed example list

### evaluate.py (agent runs this)

Execution flow:
1. Read `config.py`
2. Read `system_prompt.txt`
3. Import dataset and `compute_f1` from `prepare.py`
4. For each example: `{system_prompt + context + question}` → LLM → answer
5. Compute per-example F1, accumulate
6. Print summary to stdout:

```
---
f1_score:     0.734521
total_tokens: 48200
eval_samples: 100
model:        gpt-4.1
```

Agent extracts metric with: `grep "^f1_score:" run.log`

### system_prompt.txt (agent modifies)

Initial content:
```
You are a question answering assistant.
Given a context passage and a question, extract the answer directly from the context.
If the answer is not in the context, respond with "unanswerable".
Keep your answer concise — one phrase or sentence maximum.
```

Agent may change wording, add instructions, add few-shot examples, change format directives.

### config.py (user configures, agent does not modify)

```python
PROVIDER = "openai"           # "anthropic" | "openai"
MODEL = "gpt-4.1"
API_KEY_ENV = "OPENAI_API_KEY"
EVAL_SAMPLES = 100            # default, configurable
MAX_TOKENS = 256
TEMPERATURE = 0.0             # deterministic — reproducible results
```

`TEMPERATURE = 0.0` is critical: same prompt always produces same output, eliminating noise-driven keep/discard decisions.

---

## Data Flow

```
HuggingFace (SQuAD v2)
        ↓
  prepare.py
  └─ 100 fixed examples (seed=42)
        ↓
  evaluate.py
  ├─ reads system_prompt.txt
  ├─ reads config.py
  └─ for each example:
       context + question → LLM → answer
       compute_f1(answer, ground_truth)
        ↓
  stdout summary → run.log
        ↓
  agent reads f1_score
  └─ improved? → KEEP (git commit stays)
     worse?    → DISCARD (git reset)
     crash?    → log crash, git reset
```

---

## Logging

### results.tsv (git-untracked)

Tab-separated, 5 columns:

```
commit    f1_score   total_tokens   status    description
a1b2c3d   0.734521   48200          keep      baseline
b2c3d4e   0.761200   51000          keep      add unanswerable instruction
c3d4e5f   0.698000   44100          discard   removed context instruction
d4e5f6g   0.000000   0              crash     syntax error in prompt
```

- `f1_score`: 0.000000 for crashes
- `total_tokens`: cumulative input+output tokens, 0 for crashes
- `status`: `keep` | `discard` | `crash`

### run.log

Full stdout of last `evaluate.py` run. Per-example lines plus final summary block. Agent reads last 50 lines on crash to diagnose.

---

## analysis.ipynb

Reads `results.tsv` and produces:

1. **F1 trajectory plot** — x: experiment index, y: f1_score, color: keep/discard/crash
2. **Summary stats** — keep count, discard count, crash count, total tokens spent
3. **Best experiment** — highest F1 commit and its description
4. **Cumulative delta** — improvement from baseline to current best

Saves `progress.png` as a committable artifact.

---

## program.md Rules (summary)

- Agent reads `prepare.py`, `evaluate.py`, `system_prompt.txt`, `config.py` at setup
- First run always establishes baseline (no changes)
- Loop: modify `system_prompt.txt` → git commit → `uv run evaluate.py > run.log 2>&1` → grep F1 → keep/discard
- NEVER STOP without user interruption
- Simplicity criterion: simple prompt with +0.001 F1 beats complex prompt with +0.005 F1
- Crash handling: fix obvious errors (typos), skip fundamental failures

---

## Dependencies (pyproject.toml)

```
openai >= 1.0
anthropic >= 0.20
datasets >= 2.0
numpy >= 1.24
pandas >= 2.0
matplotlib >= 3.7
jupyter >= 1.0
```

---

## Keep/Discard Decision Threshold

Accept a change only if `f1_score` strictly improves (even by 0.001). With `TEMPERATURE=0.0` and a fixed dataset, there is no evaluation noise — any positive delta is a real improvement.
