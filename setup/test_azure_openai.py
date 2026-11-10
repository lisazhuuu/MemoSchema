import os
from openai import AzureOpenAI
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

AZURE_OPENAI_CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
AZURE_OPENAI_CHAT_API_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY")
AZURE_OPENAI_CHAT_API_VERSION = os.getenv("AZURE_OPENAI_CHAT_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

if not AZURE_OPENAI_CHAT_ENDPOINT or not AZURE_OPENAI_CHAT_API_KEY or not AZURE_OPENAI_CHAT_DEPLOYMENT:
    raise RuntimeError(
        "Missing AZURE_OPENAI_CHAT_* environment variables. "
        "Copy .env.example to .env, fill in your credentials, then load it with: "
        "set -a; source .env; set +a"
    )

chat_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
    api_key=AZURE_OPENAI_CHAT_API_KEY,
    api_version=AZURE_OPENAI_CHAT_API_VERSION,
    max_retries=int(os.getenv("MEMOSCHEMA_API_MAX_RETRIES", "2")),
    timeout=float(os.getenv("MEMOSCHEMA_API_TIMEOUT", "600")),   # SDK default timeout; override for shorter request limits.
)
