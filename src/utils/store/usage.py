import datetime

from src.config import SESSION_TTL as _SESSION_TTL
from src.utils.store.client import redis_client

# Per-day counters are kept this long, then expire on their own.
_DAILY_RETENTION = 90 * 86400


def _session_key_prefix(app_name: str, session_id: str) -> str:
    return f"usage:{app_name}:session:{session_id}"


def _daily_key_prefix(app_name: str, day: str) -> str:
    return f"usage:{app_name}:daily:{day}"


def _model_key_prefix(app_name: str, model_id: str) -> str:
    return f"usage:{app_name}:model:{model_id}"


def _models_set_key(app_name: str) -> str:
    """Set of model ids this app has ever used.

    Kept alongside the counters so the breakdown is a pipelined read of known
    keys rather than a SCAN over the whole usage keyspace.
    """
    return f"usage:{app_name}:models"


def _today() -> datetime.date:
    return datetime.datetime.now(datetime.UTC).date()


async def record_usage(
    app_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    session_id: str | None = None,
    model_id: str | None = None,
) -> None:
    pipe = redis_client.pipeline()
    pipe.incrby(f"usage:{app_name}:prompt_tokens", prompt_tokens)
    pipe.incrby(f"usage:{app_name}:completion_tokens", completion_tokens)
    pipe.incrby(f"usage:{app_name}:requests", 1)
    # Daily buckets (UTC) alongside the lifetime totals, so usage over a
    # window is answerable; each bucket expires after _DAILY_RETENTION.
    daily = _daily_key_prefix(app_name, _today().isoformat())
    pipe.incrby(f"{daily}:prompt_tokens", prompt_tokens)
    pipe.expire(f"{daily}:prompt_tokens", _DAILY_RETENTION)
    pipe.incrby(f"{daily}:completion_tokens", completion_tokens)
    pipe.expire(f"{daily}:completion_tokens", _DAILY_RETENTION)
    pipe.incrby(f"{daily}:requests", 1)
    pipe.expire(f"{daily}:requests", _DAILY_RETENTION)
    if model_id:
        # Lifetime totals per model, so cost can be attributed across a
        # registry where a token is not worth the same everywhere. Four extra
        # pipeline ops — no additional round-trip.
        prefix = _model_key_prefix(app_name, model_id)
        pipe.sadd(_models_set_key(app_name), model_id)
        pipe.incrby(f"{prefix}:prompt_tokens", prompt_tokens)
        pipe.incrby(f"{prefix}:completion_tokens", completion_tokens)
        pipe.incrby(f"{prefix}:requests", 1)
    if session_id:
        # Session counters expire with the session; each write refreshes the
        # TTL the same way session activity refreshes the session key.
        prefix = _session_key_prefix(app_name, session_id)
        pipe.incrby(f"{prefix}:prompt_tokens", prompt_tokens)
        pipe.expire(f"{prefix}:prompt_tokens", _SESSION_TTL)
        pipe.incrby(f"{prefix}:completion_tokens", completion_tokens)
        pipe.expire(f"{prefix}:completion_tokens", _SESSION_TTL)
        pipe.incrby(f"{prefix}:requests", 1)
        pipe.expire(f"{prefix}:requests", _SESSION_TTL)
    await pipe.execute()


async def get_usage(app_name: str) -> dict:
    pipe = redis_client.pipeline()
    pipe.get(f"usage:{app_name}:prompt_tokens")
    pipe.get(f"usage:{app_name}:completion_tokens")
    pipe.get(f"usage:{app_name}:requests")
    prompt, completion, requests = await pipe.execute()
    return {
        "app_name": app_name,
        "requests": int(requests or 0),
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(prompt or 0) + int(completion or 0),
    }


async def get_usage_by_model(app_name: str) -> list[dict]:
    """Per-model lifetime usage for one app, busiest first.

    Empty for an app whose traffic predates the model dimension — the totals in
    ``get_usage`` stay authoritative, so the two need not add up on a
    deployment that upgraded mid-life.
    """
    model_ids = sorted(await redis_client.smembers(_models_set_key(app_name)))  # type: ignore[misc]
    if not model_ids:
        return []
    pipe = redis_client.pipeline()
    for model_id in model_ids:
        prefix = _model_key_prefix(app_name, model_id)
        pipe.get(f"{prefix}:prompt_tokens")
        pipe.get(f"{prefix}:completion_tokens")
        pipe.get(f"{prefix}:requests")
    values = await pipe.execute()
    out: list[dict] = []
    for i, model_id in enumerate(model_ids):
        prompt, completion, requests = values[3 * i: 3 * i + 3]
        out.append({
            "model": model_id,
            "requests": int(requests or 0),
            "prompt_tokens": int(prompt or 0),
            "completion_tokens": int(completion or 0),
            "total_tokens": int(prompt or 0) + int(completion or 0),
        })
    return sorted(out, key=lambda row: row["total_tokens"], reverse=True)


async def get_daily_usage(app_name: str, days: int = 7) -> list[dict]:
    """Per-day usage for the last ``days`` UTC days, most recent first.

    Days older than _DAILY_RETENTION (or with no traffic) come back as zeros.
    """
    dates = [(_today() - datetime.timedelta(days=i)).isoformat() for i in range(days)]
    pipe = redis_client.pipeline()
    for day in dates:
        prefix = _daily_key_prefix(app_name, day)
        pipe.get(f"{prefix}:prompt_tokens")
        pipe.get(f"{prefix}:completion_tokens")
        pipe.get(f"{prefix}:requests")
    values = await pipe.execute()
    out: list[dict] = []
    for i, day in enumerate(dates):
        prompt, completion, requests = values[3 * i: 3 * i + 3]
        out.append({
            "date": day,
            "requests": int(requests or 0),
            "prompt_tokens": int(prompt or 0),
            "completion_tokens": int(completion or 0),
            "total_tokens": int(prompt or 0) + int(completion or 0),
        })
    return out


async def get_session_usage(app_name: str, session_id: str) -> dict:
    prefix = _session_key_prefix(app_name, session_id)
    pipe = redis_client.pipeline()
    pipe.get(f"{prefix}:prompt_tokens")
    pipe.get(f"{prefix}:completion_tokens")
    pipe.get(f"{prefix}:requests")
    prompt, completion, requests = await pipe.execute()
    return {
        "session_id": session_id,
        "requests": int(requests or 0),
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(prompt or 0) + int(completion or 0),
    }


async def delete_session_usage(app_name: str, session_id: str) -> None:
    prefix = _session_key_prefix(app_name, session_id)
    await redis_client.delete(
        f"{prefix}:prompt_tokens",
        f"{prefix}:completion_tokens",
        f"{prefix}:requests",
    )


async def delete_all_session_usage(app_name: str) -> None:
    keys = [key async for key in redis_client.scan_iter(f"usage:{app_name}:session:*")]
    if keys:
        await redis_client.delete(*keys)  # type: ignore[arg-type]


async def get_all_app_names() -> list[str]:
    """Returns every app_name that has a usage record."""
    app_names: set[str] = set()
    async for key in redis_client.scan_iter("usage:*:requests"):
        parts = str(key).split(":")
        if len(parts) >= 2:
            app_names.add(parts[1])
    return list(app_names)
