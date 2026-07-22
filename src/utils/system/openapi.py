"""OpenAPI schema post-processing.

FastAPI emits OpenAPI 3.1, where an ``UploadFile`` is described as
``{"type": "string", "contentMediaType": "application/octet-stream"}``. The
Swagger UI bundled with FastAPI only renders a file-picker widget for the older
``{"type": "string", "format": "binary"}`` spelling, so upload endpoints show up
as plain text boxes and can't be exercised from /swagger/docs.

Adding ``format: binary`` alongside ``contentMediaType`` keeps the document
valid 3.1 while giving Swagger UI the hint it looks for.
"""

from typing import Any, Callable

from fastapi import FastAPI

_BINARY_MEDIA_TYPE = "application/octet-stream"


def _add_binary_format(node: Any) -> None:
    """Recursively tag every binary-string schema node with ``format: binary``."""
    if isinstance(node, dict):
        if (
            node.get("type") == "string"
            and node.get("contentMediaType") == _BINARY_MEDIA_TYPE
            and "format" not in node
        ):
            node["format"] = "binary"
        for value in node.values():
            _add_binary_format(value)
    elif isinstance(node, list):
        for item in node:
            _add_binary_format(item)


def setup_openapi(app: FastAPI) -> None:
    """Wrap ``app.openapi`` so generated schemas render file inputs in Swagger UI.

    The patch mutates the dict FastAPI caches on ``app.openapi_schema``; it is
    idempotent, so the cached document stays correct across repeated calls.
    """
    original: Callable[[], dict[str, Any]] = app.openapi

    def patched_openapi() -> dict[str, Any]:
        schema = original()
        _add_binary_format(schema)
        return schema

    app.openapi = patched_openapi  # type: ignore[method-assign]
