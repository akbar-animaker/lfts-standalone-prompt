#!/usr/bin/env python3
"""
Prompt Engineering Playground — entry point.

Wraps standalone.py in a full interactive web UI so prompt engineers can:
  • Edit all prompts directly in the browser (Monaco editor)
  • Tweak pipeline config without touching code
  • Run the pipeline and watch logs stream live
  • Compare outputs and manage version history
  • Export prompts, configs, and results

Usage:
    python playground.py

Opens http://localhost:8765 automatically.
"""
import os
import sys
import time
import webbrowser
import threading

# Ensure the project root is on the path so `standalone` can be imported
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from backend.app import create_app

PORT = int(os.environ.get("PLAYGROUND_PORT", 8765))
HOST = os.environ.get("PLAYGROUND_HOST", "0.0.0.0")


def _open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{PORT}")


def main():
    print(f"\n{'='*60}")
    print("  🎬  Prompt Engineering Playground")
    print(f"{'='*60}")
    print(f"  Server : http://localhost:{PORT}")
    print(f"  Root   : {_ROOT}")
    print(f"{'='*60}\n")

    # Skip browser launch when running inside a container (no display)
    if not os.environ.get("DOCKER_ENV") and os.environ.get("DISPLAY", "") != "" or \
            sys.platform in ("darwin", "win32"):
        t = threading.Thread(target=_open_browser, daemon=True)
        t.start()

    app = create_app()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
