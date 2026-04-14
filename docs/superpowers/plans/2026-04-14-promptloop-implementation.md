# promptloop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous prompt optimization loop that iterates on `system_prompt.txt` using SQuAD v2 F1 score as the evaluation metric.

**Architecture:** `prepare.py` handles data download and F1 computation (fixed, never modified). `evaluate.py` loads the dataset, calls the LLM with the current system prompt, computes F1, and prints a structured summary. The agent modifies only `system_prompt.txt` between runs, following `program.md` instructions.

**Tech Stack:** Python 3.10+, uv, openai/anthropic SDKs, HuggingFace datasets, pandas, matplotlib, jupyter

---

## File Map

| File | Role | Agent touches? |
|------|------|---------------|
| `prepare.py` | Download SQuAD v2, `compute_f1`, `load_eval_dataset` | No |
| `evaluate.py` | LLM call loop, F1 accumulation, stdout summary | No |
| `config.py` | PROVIDER, MODEL, EVAL_SAMPLES, TEMPERATURE | No |
| `system_prompt.txt` | The prompt being optimized | Yes — only this |
| `program.md` | Agent instructions | No |
| `analysis.ipynb` | Visualize results.tsv | No |
| `.gitignore` | Ignore results.tsv, run.log, cache | No |
| `pyproject.toml` | Already created by uv init + uv add | No |

---

## Task 1: .gitignore and config.py

**Files:**
- Create: `promptloop/.gitignore`
- Create: `promptloop/config.py`

- [ ] **Step 1: Create .gitignore**

```
# Experiment outputs (untracked by design)
results.tsv
run.log
progress.png

# Cache
__pycache__/
*.pyc
.python-version

# Data cache
.cache/
```

Save to `/home/mehmetalpay/promptloop/.gitignore`

- [ ] **Step 2: Create config.py**

```python
# config.py — user configures, agent does NOT modify

PROVIDER = "openai"            # "openai" | "anthropic"
MODEL = "gpt-4.1"
API_KEY_ENV = "OPENAI_API_KEY" # env var name that holds the API key

EVAL_SAMPLES = 100             # number of SQuAD examples per evaluation run
MAX_TOKENS = 256               # max tokens in LLM response
TEMPERATURE = 0.0              # deterministic — same prompt → same output always
```

Save to `/home/mehmetalpay/promptloop/config.py`

- [ ] **Step 3: Commit**

```bash
cd /home/mehmetalpay/promptloop
git add .gitignore config.py
git commit -m "feat: add config and gitignore"
```

---

## Task 2: prepare.py — Data Download and F1

**Files:**
- Create: `promptloop/prepare.py`

- [ ] **Step 1: Create prepare.py**

```python
"""
Fixed data preparation and evaluation utilities for promptloop.
Downloads SQuAD v2 and provides compute_f1 and load_eval_dataset.

Usage:
    uv run prepare.py    # download data to ~/.cache/promptloop/

DO NOT MODIFY — this file is the fixed evaluation harness.
"""

import os
import re
import json
import string
import argparse
from collections import Counter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "promptloop")
DATASET_PATH = os.path.join(CACHE_DIR, "squad_v2.jsonl")
SEED = 42

# ---------------------------------------------------------------------------
# F1 metric (official SQuAD evaluation)
# ---------------------------------------------------------------------------

def _normalize_answer(s):
    """Lower text, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, ground_truths):
    """
    Compute token-level F1 between prediction and list of ground truth answers.
    Returns the max F1 across all ground truths (official SQuAD scoring).
    If ground_truths is empty (unanswerable), returns 1.0 if prediction
    contains 'unanswerable', else 0.0.
    """
    if not ground_truths:
        # Unanswerable question
        return 1.0 if "unanswerable" in prediction.lower() else 0.0

    best_f1 = 0.0
    pred_tokens = _normalize_answer(prediction).split()

    for gt in ground_truths:
        gt_tokens = _normalize_answer(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


# ---------------------------------------------------------------------------
# Dataset download and loading
# ---------------------------------------------------------------------------

def download_dataset():
    """Download SQuAD v2 validation split and save as JSONL."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(DATASET_PATH):
        print(f"Dataset already exists at {DATASET_PATH}")
        return

    print("Downloading SQuAD v2 validation split from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")

    with open(DATASET_PATH, "w") as f:
        for example in ds:
            row = {
                "id": example["id"],
                "title": example["title"],
                "context": example["context"],
                "question": example["question"],
                "answers": example["answers"]["text"],  # list of answer strings
            }
            f.write(json.dumps(row) + "\n")

    print(f"Saved {len(ds)} examples to {DATASET_PATH}")


def load_eval_dataset(eval_samples):
    """
    Load a fixed subset of eval_samples examples from the JSONL cache.
    Uses SEED=42 for reproducibility — same examples every run regardless of
    eval_samples value (examples are drawn from the front of a seeded shuffle).
    """
    import random

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run: uv run prepare.py"
        )

    with open(DATASET_PATH) as f:
        all_examples = [json.loads(line) for line in f]

    rng = random.Random(SEED)
    shuffled = all_examples[:]
    rng.shuffle(shuffled)
    return shuffled[:eval_samples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SQuAD v2 for promptloop")
    parser.parse_args()
    download_dataset()
    print("Done! Ready to evaluate.")
```

Save to `/home/mehmetalpay/promptloop/prepare.py`

- [ ] **Step 2: Verify download works**

```bash
cd /home/mehmetalpay/promptloop
uv run prepare.py
```

Expected output:
```
Downloading SQuAD v2 validation split from HuggingFace...
Saved 11873 examples to ~/.cache/promptloop/squad_v2.jsonl
Done! Ready to evaluate.
```

- [ ] **Step 3: Verify F1 function manually**

```bash
cd /home/mehmetalpay/promptloop
uv run python -c "
from prepare import compute_f1
print(compute_f1('in the late 1990s', ['in the late 1990s']))  # expect 1.0
print(compute_f1('the 1990s', ['in the late 1990s']))          # expect ~0.5
print(compute_f1('unanswerable', []))                          # expect 1.0
print(compute_f1('Paris', []))                                 # expect 0.0
"
```

Expected:
```
1.0
0.5
1.0
0.0
```

- [ ] **Step 4: Commit**

```bash
cd /home/mehmetalpay/promptloop
git add prepare.py
git commit -m "feat: add prepare.py with SQuAD v2 download and compute_f1"
```

---

## Task 3: system_prompt.txt — Initial Prompt

**Files:**
- Create: `promptloop/system_prompt.txt`

- [ ] **Step 1: Create system_prompt.txt**

```
You are a question answering assistant.
Given a context passage and a question, extract the answer directly from the context.
If the answer is not in the context, respond with "unanswerable".
Keep your answer concise — one phrase or sentence maximum.
```

Save to `/home/mehmetalpay/promptloop/system_prompt.txt`

- [ ] **Step 2: Commit**

```bash
cd /home/mehmetalpay/promptloop
git add system_prompt.txt
git commit -m "feat: add initial system_prompt.txt"
```

---

## Task 4: evaluate.py — LLM Evaluation Loop

**Files:**
- Create: `promptloop/evaluate.py`

- [ ] **Step 1: Create evaluate.py**

```python
"""
Promptloop evaluation script.
Loads system_prompt.txt + config.py, runs LLM on EVAL_SAMPLES SQuAD v2 examples,
computes average F1, and prints structured summary.

Usage:
    uv run evaluate.py > run.log 2>&1

Agent extracts metric with:
    grep "^f1_score:" run.log
"""

import os
import sys
import time

import config
from prepare import compute_f1, load_eval_dataset

# ---------------------------------------------------------------------------
# LLM client (provider-agnostic)
# ---------------------------------------------------------------------------

def call_llm(system_prompt, context, question):
    """
    Call LLM with system prompt + user message containing context and question.
    Returns (answer_text, tokens_used).
    """
    api_key = os.environ.get(config.API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"API key not found. Set the {config.API_KEY_ENV} environment variable."
        )

    user_message = f"Context: {context}\n\nQuestion: {question}"

    if config.PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=config.MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        answer = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens

    elif config.PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=config.MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = response.content[0].text.strip()
        tokens = response.usage.input_tokens + response.usage.output_tokens

    else:
        raise ValueError(f"Unknown provider: {config.PROVIDER!r}. Use 'openai' or 'anthropic'.")

    return answer, tokens


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    # Load system prompt
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    with open(prompt_path) as f:
        system_prompt = f.read().strip()

    print(f"[promptloop] Provider: {config.PROVIDER} | Model: {config.MODEL}")
    print(f"[promptloop] Eval samples: {config.EVAL_SAMPLES}")
    print(f"[promptloop] Loading dataset...")

    examples = load_eval_dataset(config.EVAL_SAMPLES)
    total = len(examples)

    print(f"[promptloop] Starting evaluation ({total} examples)...")
    print()

    total_f1 = 0.0
    total_tokens = 0
    t0 = time.time()

    for i, example in enumerate(examples, 1):
        context = example["context"]
        question = example["question"]
        ground_truths = example["answers"]  # list of strings, empty if unanswerable

        try:
            prediction, tokens = call_llm(system_prompt, context, question)
        except Exception as e:
            print(f"  example {i:03d}/{total} | ERROR: {e}", flush=True)
            sys.exit(1)

        f1 = compute_f1(prediction, ground_truths)
        total_f1 += f1
        total_tokens += tokens

        # Truncate prediction for display
        pred_display = prediction[:60] + "..." if len(prediction) > 60 else prediction
        gt_display = ground_truths[0] if ground_truths else "unanswerable"
        print(
            f"  example {i:03d}/{total} | f1: {f1:.2f} | "
            f"pred: {pred_display!r} | truth: {gt_display!r}",
            flush=True,
        )

    avg_f1 = total_f1 / total
    elapsed = time.time() - t0

    print()
    print("---")
    print(f"f1_score:     {avg_f1:.6f}")
    print(f"total_tokens: {total_tokens}")
    print(f"eval_samples: {total}")
    print(f"model:        {config.MODEL}")
    print(f"elapsed_sec:  {elapsed:.1f}")


if __name__ == "__main__":
    main()
```

Save to `/home/mehmetalpay/promptloop/evaluate.py`

- [ ] **Step 2: Verify script runs (dry check — no API call)**

```bash
cd /home/mehmetalpay/promptloop
uv run python -c "import evaluate; print('import OK')"
```

Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
cd /home/mehmetalpay/promptloop
git add evaluate.py
git commit -m "feat: add evaluate.py with provider-agnostic LLM loop"
```

---

## Task 5: program.md — Agent Instructions

**Files:**
- Create: `promptloop/program.md`

- [ ] **Step 1: Create program.md**

```markdown
# promptloop

Autonomous prompt optimization loop. The agent iterates on `system_prompt.txt`
to maximize F1 score on a fixed SQuAD v2 subset.

## Setup

Work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr14`).
   The branch `promptloop/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b promptloop/<tag>` from current master.
3. **Read the in-scope files**: Read these for full context:
   - `prepare.py` — fixed constants, data loading, F1 metric. Do not modify.
   - `evaluate.py` — the evaluation script. Do not modify.
   - `config.py` — provider, model, EVAL_SAMPLES. Do not modify.
   - `system_prompt.txt` — the file you modify.
4. **Verify data exists**: Check that `~/.cache/promptloop/squad_v2.jsonl` exists.
   If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create with header row only:
   `commit\tf1_score\ttotal_tokens\tstatus\tdescription`
6. **Confirm and go**.

## Experimentation

**What you CAN do:**
- Modify `system_prompt.txt` — this is the only file you edit.
  Try: different instructions, few-shot examples, output format rules,
  chain-of-thought hints, unanswerable handling, answer length constraints.

**What you CANNOT do:**
- Modify `prepare.py`, `evaluate.py`, `config.py`.
- Add new files or install packages.
- Change the evaluation harness.

**Goal: maximize f1_score.** Higher is better. Fixed dataset + TEMPERATURE=0.0
means results are fully deterministic — any improvement is real.

**Simplicity criterion**: A simpler prompt with +0.001 F1 beats a complex prompt
with +0.005 F1. Removing text and getting equal or better results is a win.

**First run**: Always run baseline first (no changes to system_prompt.txt).

## Output format

```
---
f1_score:     0.734521
total_tokens: 48200
eval_samples: 100
model:        gpt-4.1
elapsed_sec:  187.3
```

Extract the key metric:
```
grep "^f1_score:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated, NOT comma-separated):

```
commit	f1_score	total_tokens	status	description
a1b2c3d	0.734521	48200	keep	baseline
b2c3d4e	0.761200	51000	keep	add unanswerable instruction
c3d4e5f	0.698000	44100	discard	removed context instruction
d4e5f6g	0.000000	0	crash	missing quote in prompt
```

- `f1_score`: 0.000000 for crashes
- `total_tokens`: 0 for crashes
- `status`: `keep` | `discard` | `crash`

## The experiment loop

LOOP FOREVER:

1. Check current git state and last f1_score in results.tsv
2. Edit `system_prompt.txt` with one experimental change
3. `git commit -m "try: <description>"`
4. `uv run evaluate.py > run.log 2>&1`
5. `grep "^f1_score:" run.log`
6. If grep is empty → crashed. Run `tail -n 50 run.log` to diagnose.
7. Log result to results.tsv
8. If f1_score improved → KEEP (stay on this commit)
9. If f1_score same or worse → DISCARD (`git reset --hard HEAD~1`)

**NEVER STOP**: Do not ask the human whether to continue. Run until interrupted.

**Crashes**: Fix obvious errors (missing quotes, typos). Skip if fundamentally broken.
```

Save to `/home/mehmetalpay/promptloop/program.md`

- [ ] **Step 2: Commit**

```bash
cd /home/mehmetalpay/promptloop
git add program.md
git commit -m "feat: add program.md agent instructions"
```

---

## Task 6: analysis.ipynb — Results Visualization

**Files:**
- Create: `promptloop/analysis.ipynb`

- [ ] **Step 1: Create analysis.ipynb**

Create the notebook at `/home/mehmetalpay/promptloop/analysis.ipynb` with the following cells:

**Cell 1 — Imports:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
```

**Cell 2 — Load results:**
```python
df = pd.read_csv("results.tsv", sep="\t")
df.index = range(1, len(df) + 1)  # 1-based experiment index
print(f"Total experiments: {len(df)}")
print(df)
```

**Cell 3 — Summary stats:**
```python
keeps   = (df["status"] == "keep").sum()
discards = (df["status"] == "discard").sum()
crashes = (df["status"] == "crash").sum()
baseline = df[df["status"] == "keep"]["f1_score"].iloc[0] if keeps > 0 else None
best_row = df[df["status"] == "keep"].sort_values("f1_score", ascending=False).iloc[0] if keeps > 0 else None

print(f"Keep:    {keeps}")
print(f"Discard: {discards}")
print(f"Crash:   {crashes}")
if baseline and best_row is not None:
    delta = best_row["f1_score"] - baseline
    print(f"Baseline F1:  {baseline:.6f}")
    print(f"Best F1:      {best_row['f1_score']:.6f}  (commit {best_row['commit']})")
    print(f"Improvement:  +{delta:.6f}  ({delta/baseline*100:.2f}%)")
    print(f"Description:  {best_row['description']}")
print(f"Total tokens: {df['total_tokens'].sum():,}")
```

**Cell 4 — F1 trajectory plot:**
```python
color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "crash": "#95a5a6"}
colors = df["status"].map(color_map)

fig, ax = plt.subplots(figsize=(12, 5))

# Plot all points
ax.scatter(df.index, df["f1_score"], c=colors, s=80, zorder=3)

# Connect kept experiments with a line
kept = df[df["status"] == "keep"]
if len(kept) > 1:
    ax.plot(kept.index, kept["f1_score"], color="#2ecc71", linewidth=1.5,
            linestyle="--", zorder=2, alpha=0.7)

ax.set_xlabel("Experiment #")
ax.set_ylabel("F1 Score")
ax.set_title("promptloop — F1 Score by Experiment")
ax.set_xticks(df.index)
ax.set_xticklabels(df["commit"].tolist(), rotation=45, ha="right", fontsize=7)
ax.grid(axis="y", alpha=0.3)

legend_patches = [
    mpatches.Patch(color="#2ecc71", label="keep"),
    mpatches.Patch(color="#e74c3c", label="discard"),
    mpatches.Patch(color="#95a5a6", label="crash"),
]
ax.legend(handles=legend_patches, loc="lower right")

plt.tight_layout()
plt.savefig("progress.png", dpi=150)
plt.show()
print("Saved progress.png")
```

- [ ] **Step 2: Commit**

```bash
cd /home/mehmetalpay/promptloop
git add analysis.ipynb
git commit -m "feat: add analysis.ipynb for results visualization"
```

---

## Task 7: Final Verification

- [ ] **Step 1: Verify file structure**

```bash
ls /home/mehmetalpay/promptloop/
```

Expected output:
```
analysis.ipynb  config.py  docs  evaluate.py  prepare.py
program.md  pyproject.toml  system_prompt.txt  uv.lock  .gitignore
```

- [ ] **Step 2: Verify git log**

```bash
cd /home/mehmetalpay/promptloop && git log --oneline
```

Expected (newest first):
```
xxxxxxx feat: add analysis.ipynb for results visualization
xxxxxxx feat: add program.md agent instructions
xxxxxxx feat: add evaluate.py with provider-agnostic LLM loop
xxxxxxx feat: add initial system_prompt.txt
xxxxxxx feat: add prepare.py with SQuAD v2 download and compute_f1
xxxxxxx feat: add config and gitignore
xxxxxxx docs: add promptloop design spec
```

- [ ] **Step 3: Verify imports all resolve**

```bash
cd /home/mehmetalpay/promptloop
uv run python -c "
import config
import prepare
import evaluate
print('PROVIDER:', config.PROVIDER)
print('MODEL:', config.MODEL)
print('EVAL_SAMPLES:', config.EVAL_SAMPLES)
print('All imports OK')
"
```

Expected:
```
PROVIDER: openai
MODEL: gpt-4.1
EVAL_SAMPLES: 100
All imports OK
```

- [ ] **Step 4: Final commit**

```bash
cd /home/mehmetalpay/promptloop
git status  # should be clean
```
