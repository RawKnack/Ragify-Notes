import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

QDRANT_URL         = os.getenv("QDRANT_URL", ":memory:")   # defaults to local in-memory
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY", "")

# Warn but don't crash — only the key you're using needs to be set
if not OPENROUTER_API_KEY:
    print("⚠️  OPENROUTER_API_KEY not set — needed for GPT-4o-mini and embeddings")
