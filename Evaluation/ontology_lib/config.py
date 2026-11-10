# Output directory
DEFAULT_OUT_DIR = "./ontology_eval_results"

# LLM
DEFAULT_MODEL_DEPLOYMENT = "gpt-4o"
MAX_COMPLETION_TOKENS = 3500
LLM_RETRIES = 3
LLM_RETRY_BACKOFF_SEC = 2.0
LLM_TEMPERATURE = 0.0   # Use temperature 0 for evaluation.

# Scoring
# Allowed match-strength labels the judge LLM may return.
MATCH_LABELS = ("strong", "partial", "none")

# Partial-match weight for strict one-to-one scoring.
# Lenient scoring gives partial matches full credit.
PARTIAL_CREDIT_WEIGHT = 0.5

# Prompt shaping
# Limit aliases and member fields included in the judge prompt.
MAX_ALIASES_IN_PROMPT = 20
MAX_MEMBER_FIELDS_IN_PROMPT = 10
