# config.py — user configures, agent does NOT modify

PROVIDER = "openai"            # "openai" | "anthropic"
MODEL = "gpt-4.1"
API_KEY_ENV = "OPENAI_API_KEY" # env var name that holds the API key

EVAL_SAMPLES = 100             # number of SQuAD examples per evaluation run
MAX_TOKENS = 256               # max tokens in LLM response
TEMPERATURE = 0.0              # deterministic — same prompt → same output always
