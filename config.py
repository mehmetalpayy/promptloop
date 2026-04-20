# config.py — user configures, agent does NOT modify

PROVIDER = "openai"            # "openai" | "anthropic"
MODEL = "gpt-4.1"
API_KEY_ENV = "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" # env var name that holds the API key

EVAL_SAMPLES = 100             # number of SQuAD examples per evaluation run
MAX_TOKENS = 256               # max tokens in LLM response
TEMPERATURE = 0.0              # deterministic — same prompt → same output always
