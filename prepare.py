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
                "answers": example["answers"]["text"],  # list of strings
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
    parser = argparse.ArgumentParser(description="Download SQuAD v2 for promptloop")  # noqa: E501
    parser.parse_args()
    download_dataset()
    print("Done! Ready to evaluate.")
