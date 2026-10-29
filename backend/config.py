"""
Configuration module for AI Document Intelligence.
Loads environment variables and exposes typed settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Search for .env in backend directory, parent directory, or current working directory
_backend_dir = Path(__file__).resolve().parent
_candidates = [
    _backend_dir / ".env",
    _backend_dir.parent / ".env",
    Path.cwd() / ".env"
]

for _env_candidate in _candidates:
    if _env_candidate.exists():
        load_dotenv(dotenv_path=_env_candidate, override=True)
        break
else:
    load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"

# Aliases for convenient importing across modules
CHAT_MODEL: str = OPENAI_CHAT_MODEL
EMBEDDING_MODEL: str = OPENAI_EMBEDDING_MODEL

# Validate that OPENAI_API_KEY is present
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Please provide a valid OpenAI API key in your .env file "
        "or environment variables (e.g., OPENAI_API_KEY=sk-...)."
    )
