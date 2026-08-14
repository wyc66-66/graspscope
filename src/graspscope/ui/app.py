from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from graspscope import __version__
from graspscope.errors import GraspScopeError
from graspscope.paths import safe_under_root
from graspscope.ui import repo_root

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="GraspScope", version=__version__)
    root = repo_root()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "service": "graspscope"}

    @app.get("/api/graspscope/frontier.json")
    def graspscope_frontier() -> dict[str, Any]:
        """GraspScope closed-loop results (frontier + gate + scenes)."""
        path = root / "data" / "grasp_closedloop" / "frontier.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="grasp results missing — run scripts/grasp_run_closedloop.py")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/graspscope/scene")
    def graspscope_scene(f: str) -> FileResponse:
        """Serve a synthetic grasp scene image by filename."""
        img_dir = root / "data" / "grasp_synth" / "images"
        try:
            target = safe_under_root(Path(f), img_dir)
        except GraspScopeError as e:
            raise HTTPException(status_code=e.http_status, detail=str(e)) from e
        if not target.is_file():
            raise HTTPException(status_code=404, detail="scene image not found")
        return FileResponse(target, media_type="image/jpeg")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/graspscope")
    def graspscope_index() -> FileResponse:
        index_path = STATIC_DIR / "graspscope.html"
        if not index_path.is_file():
            raise HTTPException(status_code=500, detail="GraspScope UI static file missing")
        return FileResponse(index_path)

    @app.get("/")
    def index() -> FileResponse:
        index_path = STATIC_DIR / "graspscope.html"
        if not index_path.is_file():
            raise HTTPException(status_code=500, detail="UI static files missing")
        return FileResponse(index_path)

    return app
