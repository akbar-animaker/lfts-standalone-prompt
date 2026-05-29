"""
Pipeline runner: spawns one isolated subprocess per run.
Multiple users can run concurrently — no shared module state, no file collision.
Each run gets:
  • Its own Python process (isolated standalone.py import)
  • A config+prompt snapshot taken at start time
  • A unique output directory: storage/runs/{run_id}/
"""
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.services.storage import (
    get_prompts, get_config, save_run_result, save_version, get_run_result
)

_BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_worker.py")

# ── Run registry ──────────────────────────────────────────────────────────────
_runs: Dict[str, "RunState"] = {}
_active_run_id: Optional[str] = None   # most-recently-started run still executing
_state_lock = threading.Lock()


class RunState:
    def __init__(self, run_id: str, video_path: str = ""):
        self.run_id     = run_id
        self.video_path = video_path
        self.status     = "starting"
        self.logs: list = []
        self.result: Optional[Dict] = None
        self.error: Optional[str]   = None
        self.log_queue: queue.Queue = queue.Queue()
        self.start_time = datetime.utcnow().isoformat()
        self.end_time: Optional[str] = None


def reload_standalone():
    """No-op in subprocess mode — each run gets a fresh import automatically."""
    pass


# ── Log persistence ───────────────────────────────────────────────────────────

def _save_logs(run_state: RunState):
    """Write accumulated logs to storage/runs/{run_id}/run.log."""
    try:
        log_dir = os.path.join(_BASE, "storage", "runs", run_state.run_id)
        os.makedirs(log_dir, exist_ok=True)
        lines = []
        for e in run_state.logs:
            ts  = e.get("timestamp", "")
            lvl = e.get("level", "INFO")
            msg = e.get("message", "")
            lines.append(f"[{ts}] [{lvl}] {msg}")
        with open(os.path.join(log_dir, "run.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass


# ── Worker output reader (runs in a daemon thread) ────────────────────────────

def _read_worker(run_state: RunState, proc: subprocess.Popen):
    global _active_run_id

    def _push(entry: dict):
        run_state.logs.append(entry)
        run_state.log_queue.put(entry)

    def _log(level: str, msg: str):
        _push({"type": "log", "level": level, "message": msg,
               "timestamp": datetime.utcnow().isoformat()})

    try:
        run_state.status = "running"

        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                _log("INFO", line)
                continue

            if msg.get("type") == "done":
                run_state.end_time = datetime.utcnow().isoformat()

                if msg.get("status") == "success":
                    result = msg.get("result", {})
                    run_state.status = "success"
                    run_state.result = result
                    summary = {
                        "sequential_count":    len(result.get("sequential_clips", [])),
                        "non_sequential_count": len(result.get("non_sequential_shorts", [])),
                        "video_duration":      result.get("video_duration", 0),
                        "category":            result.get("video_category", ""),
                        "total_tokens":        result.get("token_usage", {}).get("total_tokens", 0),
                    }
                    save_run_result(run_state.run_id, {
                        "run_id":     run_state.run_id,
                        "status":     "success",
                        "start_time": run_state.start_time,
                        "end_time":   run_state.end_time,
                        "summary":    summary,
                        "result":     result,
                        "video_path": run_state.video_path,
                    })
                    _save_logs(run_state)
                    run_state.log_queue.put({"type": "done", "status": "success", "summary": summary})
                else:
                    run_state.status = "error"
                    run_state.error  = msg.get("error", "Unknown error")
                    _save_logs(run_state)
                    run_state.log_queue.put({"type": "done", "status": "error", "error": run_state.error})
                break
            else:
                # Regular log entry from the worker
                entry = {**msg, "timestamp": datetime.utcnow().isoformat()}
                _push(entry)

        proc.wait()

        # Guard against worker dying before sending a "done" message
        if run_state.status == "running":
            run_state.status   = "error"
            run_state.error    = f"Worker exited unexpectedly (code {proc.returncode})"
            run_state.end_time = datetime.utcnow().isoformat()
            run_state.log_queue.put({"type": "done", "status": "error", "error": run_state.error})

    except Exception as exc:
        run_state.status   = "error"
        run_state.error    = str(exc)
        run_state.end_time = datetime.utcnow().isoformat()
        run_state.log_queue.put({"type": "done", "status": "error", "error": str(exc)})
    finally:
        with _state_lock:
            if _active_run_id == run_state.run_id:
                _active_run_id = None


# ── Public API ────────────────────────────────────────────────────────────────

def start_run(
    transcript_path: str,
    video_path: Optional[str],
    clip_mode: str = "both",
    save_version_flag: bool = False,
    prompt_overrides: Optional[Dict[str, str]] = None,
    code_override: Optional[str] = None,
) -> str:
    global _active_run_id

    run_id     = uuid.uuid4().hex[:8]
    run_state  = RunState(run_id, video_path=video_path or "")

    with _state_lock:
        _runs[run_id] = run_state
        _active_run_id = run_id

    # Snapshot config + prompts at start time — each run is independent
    prompts = prompt_overrides if prompt_overrides is not None else get_prompts()
    config  = get_config()

    if save_version_flag:
        save_version(note=f"Auto-save before run {run_id}")

    # Per-run isolated output directory
    output_dir = os.path.join(_BASE, "storage", "runs", run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Serialise run args to a file in the run directory
    args_path = os.path.join(output_dir, "run_args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id":           run_id,
            "transcript_path":  transcript_path,
            "video_path":       video_path or "",
            "clip_mode":        clip_mode,
            "config":           config,          # full unmasked config snapshot
            "prompt_overrides": prompts,         # resolved: draft OR stored
            "code_override":    code_override,
            "output_dir":       output_dir,
        }, f)

    # Spawn isolated worker process
    proc = subprocess.Popen(
        [sys.executable, _WORKER, args_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout stream
        text=True,
        bufsize=1,
        cwd=_BASE,
    )

    thread = threading.Thread(
        target=_read_worker,
        args=(run_state, proc),
        daemon=True,
    )
    thread.start()
    return run_id


def get_run(run_id: str) -> Optional[RunState]:
    return _runs.get(run_id)


def get_active_run_id() -> Optional[str]:
    return _active_run_id


def get_run_status(run_id: str) -> Optional[Dict]:
    state = _runs.get(run_id)
    if state:
        return {
            "run_id":     state.run_id,
            "status":     state.status,
            "start_time": state.start_time,
            "end_time":   state.end_time,
            "error":      state.error,
            "log_count":  len(state.logs),
            "has_result": state.result is not None,
        }
    saved = get_run_result(run_id)
    if saved:
        return {
            "run_id":     saved.get("run_id"),
            "status":     saved.get("status"),
            "start_time": saved.get("start_time"),
            "end_time":   saved.get("end_time"),
            "log_count":  0,
            "has_result": True,
        }
    return None
