# backend/utils/redis_client.py
import os
import json
from dotenv import load_dotenv
import redis

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# decode_responses True so we read/write str
r = redis.from_url(REDIS_URL, decode_responses=True)

def _key(k: str) -> str:
    return f"fraud:{k}"

def get_cache(key: str):
    try:
        v = r.get(_key(key))
        if not v:
            return None
        return json.loads(v)
    except Exception:
        return None

def set_cache(key: str, value, ttl: int = 3600):
    try:
        r.set(_key(key), json.dumps(value), ex=ttl)
        return True
    except Exception:
        return False
