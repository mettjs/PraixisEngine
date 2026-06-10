import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["Admin UI"])

_ROUTES_DIR    = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES_DIR = os.path.normpath(os.path.join(_ROUTES_DIR, "..", "admin_panel"))

# Mounted as a StaticFiles app in main.py (mounts can't live on an APIRouter
# that gets include_router'd).
STATIC_DIR = os.path.normpath(os.path.join(_ROUTES_DIR, "..", "admin_panel", "static"))

templates = Jinja2Templates(directory=_TEMPLATES_DIR)


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def admin_ui(request: Request):
    return templates.TemplateResponse(request, "base.html")
