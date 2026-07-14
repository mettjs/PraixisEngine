import json
import hashlib
import datetime
import secrets
from src.utils.store.client import redis_client

# Every issued key carries this prefix. The rate limiter buckets on it, so key
# generation and the limiter must always agree on the exact string.
KEY_PREFIX = "praixis_"


def new_api_key() -> str:
    """A fresh raw API key carrying the issued prefix."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(raw_key: str) -> str:
    """The canonical digest of a raw key: it names the ``apikey:{hash}`` store
    entry (and is the ``key_hash`` shown by admin endpoints), and the rate
    limiter uses it to keep raw keys out of its bucket names."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def store_api_key(full_key: str, app_name: str) -> None:
    value = json.dumps({
        "app_name": app_name,
        "key_preview": full_key[:14] + "...",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    })
    await redis_client.set(f"apikey:{hash_api_key(full_key)}", value)


async def lookup_api_key(full_key: str) -> str | None:
    data = await redis_client.get(f"apikey:{hash_api_key(full_key)}")
    if not isinstance(data, str):
        return None
    try:
        return json.loads(data).get("app_name")
    except json.JSONDecodeError:
        return None


async def get_api_key_entry(key_hash: str) -> dict | None:
    """The stored metadata for a key hash (``app_name``, ``key_preview``,
    ``created_at``), or None when the hash is unknown."""
    data = await redis_client.get(f"apikey:{key_hash}")
    if not isinstance(data, str):
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


async def list_all_api_keys() -> list[dict]:
    keys = [key async for key in redis_client.scan_iter("apikey:*")]
    if not keys:
        return []
    values = await redis_client.mget(*keys)
    entries: list[dict] = []
    for redis_key, raw in zip(keys, values):
        if not isinstance(raw, str):
            continue
        try:
            data = json.loads(raw)
            entries.append({
                "app_name": data.get("app_name"),
                "key_preview": data.get("key_preview"),
                "created_at": data.get("created_at"),
                "key_hash": str(redis_key).split(":", 1)[1],
            })
        except (json.JSONDecodeError, AttributeError):
            pass
    return entries


async def remove_api_key_by_hash(key_hash: str) -> bool:
    return await redis_client.delete(f"apikey:{key_hash}") > 0  # type: ignore[operator]
