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

# Validate OPENAI_API_KEY: if missing in environment, default to placeholder
# so the application can boot and serve the web UI gracefully
if not OPENAI_API_KEY:
    import logging
    logging.getLogger("docintel.config").warning(
        "OPENAI_API_KEY is missing. Defaulting to placeholder. "
        "Add OPENAI_API_KEY in Render Environment Variables or .env to enable live AI."
    )
    OPENAI_API_KEY = "your_api_key_here"
