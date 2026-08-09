"""Request-side model resolution.

Turns the optional ``model`` a caller sends into a validated
:class:`~src.models.registry.ModelSpec`, or a 400 naming the models it may
actually use. Pass-through of arbitrary model names is never allowed: it would
let any client reach any model behind the proxy and defeat both the per-model
pools and the per-key allowlist.

Resolution order, first hit wins:

1. the ``model`` on the request
2. the model the session is already bound to
3. the API key's ``default_model``
4. the registry's ``default``

— all filtered by the key's allowlist, so a key can never reach outside it.
"""
from fastapi import HTTPException

from src.models.registry import ModelSpec, UnknownModelError, resolve_model
from src.utils.store.sessions import get_session_model


def allowed_models(caller_entry: dict | None) -> list[str] | None:
    """The key's allowlist, or None when it is unrestricted."""
    return (caller_entry or {}).get("models") or None


def resolve_model_or_400(
    requested: str | None,
    caller_entry: dict | None = None,
    session_model: str | None = None,
) -> ModelSpec:
    """The model for this request.

    A ``session_model`` that the key may no longer use is treated as absent
    rather than as an error: the caller did not ask for it on this request, and
    re-scoping a key should downgrade an old session, not break it. An explicit
    ``requested`` model outside the allowlist is always a 400, and reads
    exactly like an unknown one — a caller is never told a model exists that it
    may not use.
    """
    entry = caller_entry or {}
    allowed = allowed_models(entry)
    if requested:
        try:
            return resolve_model(requested, allowed=allowed)
        except UnknownModelError as e:
            raise HTTPException(status_code=400, detail=str(e))
    if session_model:
        try:
            return resolve_model(session_model, allowed=allowed)
        except UnknownModelError:
            pass
    try:
        return resolve_model(None, allowed=allowed, key_default=entry.get("default_model"))
    except UnknownModelError:
        raise HTTPException(
            status_code=400,
            detail="This API key is scoped to models that are no longer configured.",
        )


async def resolve_request_model(
    requested: str | None,
    app_name: str,
    caller_entry: dict | None = None,
    session_id: str | None = None,
) -> ModelSpec:
    """:func:`resolve_model_or_400`, having first looked up the session binding.

    The binding read is skipped entirely when the request names a model or has
    no session — the common single-model case costs no extra round-trip.
    """
    session_model = None
    if not requested and session_id:
        session_model = await get_session_model(app_name=app_name, session_id=session_id)
    return resolve_model_or_400(requested, caller_entry, session_model)
