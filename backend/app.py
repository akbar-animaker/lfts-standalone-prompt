"""
FastAPI application factory.
"""
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router

_FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Prompt Engineering Playground",
        description="Interactive UI for the Video Clipping Pipeline",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(os.path.join(_FRONTEND, "index.html"))

    return app
