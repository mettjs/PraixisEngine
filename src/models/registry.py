"""The model registry: every LLM this deployment may talk to.

The engine addresses models by a short registry ``id`` (``fast``, ``smart``),
never by the raw model name a backend serves. An optional ``models.yaml`` at
the project root declares those ids; **when the file is absent the existing
``AI_API_URL`` / ``AI_API_KEY`` / ``MODEL_NAME`` / ``CONTEXT_WINDOW`` env vars
synthesize a single ``default`` entry**, so a single-model deployment upgrades
with no configuration change and identical behavior.

Everything is parsed, expanded and validated once at import, in the style of
``config.py``: a malformed file, an unknown ``default``, a duplicate id or a
role pointing at a missing model is a hard error at startup rather than a 500
on the first request that happens to hit it.

Internal LLM work (query reformulation, history compaction, background question
generation) resolves by *role* instead of by the caller's model, so routing a
user's chat to an expensive model does not silently bill the cheap chores to it
too.
"""
import errno
import os
import re
import tempfile
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from src.utils.system.logger import logger

from src.config import (
    AI_API_KEY as _AI_API_KEY,
    AI_API_URL as _AI_API_URL,
    CONTEXT_WINDOW as _CONTEXT_WINDOW,
    MODEL_NAME as _MODEL_NAME,
    MODELS_FILE as _MODELS_FILE,
)

# Same charset as ``collection_name``. Deliberately conservative: an id travels
# in the ``[MODEL:<id>]`` stream marker, and this alphabet keeps that marker
# unambiguous without the percent-escaping ``[SOURCES:...]`` needs.
_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,63}$")

# ``${VAR}`` only — never shell evaluation. Used for secrets that must not be
# committed alongside the registry.
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# The pool every model draws from unless it says otherwise: the existing
# ``gpu:slots`` bucket, so no Redis key changes on deploy.
DEFAULT_POOL = "default"
# A model served remotely consumes none of this deployment's GPU.
NO_POOL = "none"

# Internal LLM work resolves through these; each falls back to the registry
# default when the file does not map it.
ROLES = ("utility", "background")

# Request kwargs the engine owns. A models.yaml `params` block that sets one of
# these would collide with the call site's own argument.
_RESERVED_PARAMS = frozenset({"model", "messages", "stream", "stream_options"})

_SYNTHESIZED_ID = "default"


class ModelConfigError(RuntimeError):
    """Raised at import when ``models.yaml`` is unusable. Aborts startup."""


class UnknownModelError(ValueError):
    """Raised when a request names a model it may not use. Maps to a 400."""

    def __init__(self, requested: str, permitted: list[str]) -> None:
        self.requested = requested
        self.permitted = permitted
        super().__init__(
            f"Unknown model '{requested}'. Available models: {', '.join(permitted) or '(none)'}."
        )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One resolved registry entry — everything a call site needs to issue a
    request: which backend, which model name, how much context it has, and
    which concurrency pool it draws from."""

    id: str
    model: str
    api_url: str
    api_key: str
    context_window: int
    pool: str = DEFAULT_POOL
    params: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def uses_gpu(self) -> bool:
        """False for remote models, which take no local slot."""
        return self.pool != NO_POOL


@dataclass(frozen=True, slots=True)
class Registry:
    models: Mapping[str, ModelSpec]
    default_id: str
    roles: Mapping[str, str]
    pools: Mapping[str, int]

    def get(self, model_id: str) -> ModelSpec:
        spec = self.models.get(model_id)
        if spec is None:
            raise UnknownModelError(model_id, list(self.models))
        return spec

    @property
    def default(self) -> ModelSpec:
        return self.models[self.default_id]


def _expand(value: str, where: str) -> str:
    """Expands ``${VAR}`` references against the environment.

    An unset variable is a hard error: silently sending an empty API key would
    surface much later as an opaque 401 from the backend.
    """
    def _sub(match: re.Match) -> str:
        name = match.group(1)
        resolved = os.getenv(name)
        if resolved is None:
            raise ModelConfigError(
                f"{where}: environment variable '{name}' is referenced but not set."
            )
        return resolved

    return _VAR_RE.sub(_sub, value)


def _require_str(entry: dict, key: str, where: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelConfigError(f"{where}: '{key}' is required and must be a non-empty string.")
    return _expand(value.strip(), where)


def _optional_str(entry: dict, key: str, where: str, default: str) -> str:
    value = entry.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ModelConfigError(f"{where}: '{key}' must be a string.")
    # An empty api_key is legitimate (most local backends ignore it), so this
    # allows blank strings where _require_str does not.
    return _expand(value, where)


def _positive_int(value: Any, where: str, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelConfigError(f"{where}: '{key}' must be a positive integer.")
    return value


def _build_spec(entry: Any, index: int) -> ModelSpec:
    where = f"models[{index}]"
    if not isinstance(entry, dict):
        raise ModelConfigError(f"{where}: each model entry must be a mapping.")

    model_id = _require_str(entry, "id", where)
    if not _ID_RE.match(model_id):
        raise ModelConfigError(
            f"{where}: id '{model_id}' must match {_ID_RE.pattern} "
            "(letters, digits, '_' and '-', 1-63 chars)."
        )
    where = f"models['{model_id}']"

    pool = _optional_str(entry, "pool", where, DEFAULT_POOL).strip() or DEFAULT_POOL
    if pool != NO_POOL and not _ID_RE.match(pool):
        raise ModelConfigError(f"{where}: pool '{pool}' must match {_ID_RE.pattern} or be 'none'.")

    params = entry.get("params", {})
    if not isinstance(params, dict):
        raise ModelConfigError(f"{where}: 'params' must be a mapping.")
    # params is splatted into chat.completions.create alongside kwargs the call
    # sites set themselves, so a reserved key is a TypeError on every request to
    # this model. Catch it here, where the message can name the file.
    reserved = sorted(set(params) & _RESERVED_PARAMS)
    if reserved:
        raise ModelConfigError(
            f"{where}: 'params' must not set {', '.join(reserved)} — "
            "the engine sets those per request."
        )

    unknown = set(entry) - {"id", "model", "api_url", "api_key", "context_window", "pool", "params"}
    if unknown:
        raise ModelConfigError(f"{where}: unknown field(s): {', '.join(sorted(unknown))}.")

    return ModelSpec(
        id=model_id,
        model=_require_str(entry, "model", where),
        api_url=_optional_str(entry, "api_url", where, _AI_API_URL),
        api_key=_optional_str(entry, "api_key", where, _AI_API_KEY),
        context_window=(
            _CONTEXT_WINDOW
            if entry.get("context_window") is None
            else _positive_int(entry["context_window"], where, "context_window")
        ),
        pool=pool,
        params=MappingProxyType(dict(params)),
    )


def _synthesized() -> Registry:
    """The zero-configuration registry: one entry built from the env vars a
    single-model deployment already sets."""
    spec = ModelSpec(
        id=_SYNTHESIZED_ID,
        model=_MODEL_NAME,
        api_url=_AI_API_URL,
        api_key=_AI_API_KEY,
        context_window=_CONTEXT_WINDOW,
    )
    return Registry(
        models=MappingProxyType({spec.id: spec}),
        default_id=spec.id,
        roles=MappingProxyType({role: spec.id for role in ROLES}),
        pools=MappingProxyType({}),
    )


def build_registry(raw: dict | None) -> Registry:
    """Validates a parsed ``models.yaml`` body into a :class:`Registry`.

    ``None`` (no file) yields the synthesized single-model registry. Every
    other failure mode raises :class:`ModelConfigError`.
    """
    if raw is None:
        return _synthesized()
    if not isinstance(raw, dict):
        raise ModelConfigError("models.yaml: top level must be a mapping.")

    unknown = set(raw) - {"default", "roles", "pools", "models"}
    if unknown:
        raise ModelConfigError(f"models.yaml: unknown top-level key(s): {', '.join(sorted(unknown))}.")

    entries = raw.get("models")
    if not isinstance(entries, list) or not entries:
        raise ModelConfigError("models.yaml: 'models' must be a non-empty list.")

    models: dict[str, ModelSpec] = {}
    for index, entry in enumerate(entries):
        spec = _build_spec(entry, index)
        if spec.id in models:
            raise ModelConfigError(f"models.yaml: duplicate model id '{spec.id}'.")
        models[spec.id] = spec

    # 'default' is optional: with a single-entry file, or an obvious first
    # entry, spelling it out adds nothing.
    default_id = raw.get("default")
    if default_id is None:
        default_id = next(iter(models))
    elif not isinstance(default_id, str) or default_id not in models:
        raise ModelConfigError(
            f"models.yaml: 'default' must name a declared model; got '{default_id}'."
        )

    raw_roles = raw.get("roles") or {}
    if not isinstance(raw_roles, dict):
        raise ModelConfigError("models.yaml: 'roles' must be a mapping.")
    unknown_roles = set(raw_roles) - set(ROLES)
    if unknown_roles:
        raise ModelConfigError(
            f"models.yaml: unknown role(s): {', '.join(sorted(unknown_roles))}. "
            f"Supported roles: {', '.join(ROLES)}."
        )
    roles: dict[str, str] = {}
    for role in ROLES:
        target = raw_roles.get(role, default_id)
        if not isinstance(target, str) or target not in models:
            raise ModelConfigError(
                f"models.yaml: role '{role}' points at undeclared model '{target}'."
            )
        roles[role] = target

    raw_pools = raw.get("pools") or {}
    if not isinstance(raw_pools, dict):
        raise ModelConfigError("models.yaml: 'pools' must be a mapping.")
    pools: dict[str, int] = {}
    for name, size in raw_pools.items():
        if not isinstance(name, str) or not _ID_RE.match(name) or name == NO_POOL:
            raise ModelConfigError(f"models.yaml: pool name '{name}' is not a valid pool id.")
        pools[name] = _positive_int(size, f"pools['{name}']", "size")

    return Registry(
        models=MappingProxyType(models),
        default_id=default_id,
        roles=MappingProxyType(roles),
        pools=MappingProxyType(pools),
    )


def load_registry(path: str) -> Registry:
    """Reads ``path`` if it exists and builds the registry from it."""
    if not os.path.isfile(path):
        return build_registry(None)
    try:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as e:
        raise ModelConfigError(f"Could not parse '{path}': {e}") from e
    except OSError as e:
        raise ModelConfigError(f"Could not read '{path}': {e}") from e
    # An empty file is treated as no file: the env vars still describe a
    # working single-model deployment.
    return build_registry(raw if raw is not None else None)


def read_registry_file(path: str | None = None) -> dict | None:
    """The raw parsed contents of the registry file, or None when absent.

    This is what an editor reads: the file **as written**, before defaults are
    filled in. Handing back the resolved :class:`Registry` instead would make a
    round-trip through the admin panel rewrite every entry with values the
    author deliberately left to the env vars.
    """
    path = path or _MODELS_FILE
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except (yaml.YAMLError, OSError) as e:
        raise ModelConfigError(f"Could not read '{path}': {e}") from e


def write_registry_file(raw: dict | None, path: str | None = None) -> None:
    """Validates ``raw`` and writes it to ``path``, or deletes the file for None.

    Validation runs first and on the *whole* document, so a write can never
    leave a registry the engine would refuse to boot with — the failure lands
    on the editor, not on the next restart.

    The write prefers the atomic form (temp file in the same directory, renamed
    into place) so a crash mid-write cannot truncate a working registry. That is
    impossible when the target is a **bind-mounted file**, which is exactly how
    Docker supplies this one: the path is its own mountpoint, and renaming over
    it fails with EBUSY no matter who owns it. Only *that* failure falls back to
    writing in place, and only after the payload has already been written to a
    temp file successfully — so a full disk or a permission problem fails before
    the existing file is touched, rather than truncating it. If the in-place
    write fails part-way, the previous contents are restored.

    Nothing here touches the live :data:`_REGISTRY`: the running process keeps
    serving the configuration it started with, and the pools derived from it
    stay consistent. The new file takes effect on restart.
    """
    path = path or _MODELS_FILE
    if raw is None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return

    build_registry(raw)  # raises ModelConfigError before anything is written

    payload = _REGISTRY_HEADER + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".models.", suffix=".yaml.tmp")
    try:
        # A failure here (no space, no permission) leaves the existing file
        # untouched, which is the whole point of writing elsewhere first.
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        try:
            os.replace(tmp, path)
            return
        except OSError as e:
            if e.errno not in _RENAME_UNSUPPORTED:
                raise
            # The target is a mountpoint (Docker binds models.yaml as its own),
            # so it can only be written through, never replaced.
            _overwrite_in_place(payload, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# Renaming onto a bind-mounted file is EBUSY on Linux; a cross-device target
# (EXDEV) cannot be renamed onto either. Every other errno is a real failure and
# must not silently become an in-place write.
_RENAME_UNSUPPORTED = frozenset({errno.EBUSY, errno.EXDEV, errno.EPERM})


def _overwrite_in_place(payload: str, path: str) -> None:
    """Last resort for a target that cannot be renamed onto.

    Truncating is unavoidable here, so the previous contents are held in memory
    and restored if the write does not complete — an empty models.yaml reads as
    "no registry", which would silently drop every declared model on the next
    restart.
    """
    try:
        with open(path, "rb") as handle:
            previous = handle.read()
    except OSError:
        previous = None

    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload)
    except OSError:
        if previous is not None:
            try:
                with open(path, "wb") as handle:
                    handle.write(previous)
            except OSError:
                logger.error(
                    f"Failed to write '{path}' and could not restore its previous "
                    f"contents; the registry file may be incomplete."
                )
        raise


def registry_file_state(path: str | None = None) -> dict:
    """One read of the registry file, answering everything an editor needs.

    Returns ``{"file", "error", "matches_running", "writable"}``:

    * ``file`` — the document as written, or None when there is no file.
    * ``error`` — why it could not be read/parsed, when that is the case. This
      is deliberately distinct from "no file": an unreadable file must not look
      like an absent one, or an editor would happily overwrite something
      recoverable.
    * ``matches_running`` — False when the file has changed since this process
      started, i.e. what is served is not what the file says.
    * ``writable`` — whether a save could succeed at all, so the UI can say so
      up front instead of discovering it on a 500.

    Parsed once: the callers used to read and parse the same file twice per
    request, on the event loop.
    """
    path = path or _MODELS_FILE
    state = {"file": None, "error": None, "matches_running": True, "writable": _is_writable(path)}
    try:
        raw = read_registry_file(path)
    except ModelConfigError as e:
        # Unreadable: the running registry is whatever it started with, and we
        # cannot say whether the file agrees, so assume it does not.
        return {**state, "error": str(e), "matches_running": False}
    state["file"] = raw
    try:
        state["matches_running"] = build_registry(raw) == _REGISTRY
    except ModelConfigError as e:
        state["error"] = str(e)
        state["matches_running"] = False
    return state


def _is_writable(path: str) -> bool:
    """Whether ``path`` could be written — the file itself, or its directory
    when the file does not exist yet."""
    if os.path.exists(path):
        return os.access(path, os.W_OK)
    return os.access(os.path.dirname(os.path.abspath(path)) or ".", os.W_OK)


_REGISTRY_HEADER = """\
# models.yaml — the LLM registry. Written by the Praixis admin panel.
#
# Hand edits are fine; the panel round-trips whatever is here. Changes take
# effect when the engine restarts. See models.yaml.example for every field.
"""

_REGISTRY: Registry = load_registry(_MODELS_FILE)


def default_model() -> ModelSpec:
    """The registry's default — what a request that names no model gets."""
    return _REGISTRY.default


def list_models() -> list[ModelSpec]:
    return list(_REGISTRY.models.values())


def resolve_role(role: str) -> ModelSpec:
    """The model that handles an internal chore (``utility``/``background``).

    Resolved independently of the caller's model on purpose: cheap work stays
    on the cheap model even when a request routes its answer elsewhere.
    """
    return _REGISTRY.models[_REGISTRY.roles[role]]


def pool_sizes() -> Mapping[str, int]:
    """Declared per-pool concurrency budgets (Phase 2 sizes its buckets here)."""
    return _REGISTRY.pools


def resolve_model(
    requested: str | None = None,
    *,
    allowed: list[str] | None = None,
    key_default: str | None = None,
) -> ModelSpec:
    """Applies the resolution order: explicit request → the API key's default →
    the registry default.

    ``allowed`` is the caller's allowlist (``None`` = every model). An id
    outside it is rejected exactly like an unknown one — the caller is never
    told a model exists that it may not use.
    """
    permitted = [m for m in _REGISTRY.models if allowed is None or m in allowed]
    if requested:
        if requested not in permitted:
            raise UnknownModelError(requested, permitted)
        return _REGISTRY.models[requested]
    if key_default and key_default in permitted:
        return _REGISTRY.models[key_default]
    if _REGISTRY.default_id in permitted:
        return _REGISTRY.default
    if not permitted:
        raise UnknownModelError("", permitted)
    return _REGISTRY.models[permitted[0]]
