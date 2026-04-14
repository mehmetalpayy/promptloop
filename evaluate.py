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
        raise ValueError(
            f"Unknown provider: {config.PROVIDER!r}. Use 'openai' or 'anthropic'."
        )

    return answer, tokens


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def main():
    # Load system prompt
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_prompt.txt")
    with open(prompt_path) as f:
        system_prompt = f.read().strip()

    print(f"[promptloop] Provider: {config.PROVIDER} | Model: {config.MODEL}")
    print(f"[promptloop] Eval samples: {config.EVAL_SAMPLES}")
    print("[promptloop] Loading dataset...")

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
