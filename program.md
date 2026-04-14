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
