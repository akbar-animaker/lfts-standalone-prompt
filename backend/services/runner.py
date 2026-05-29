"""
Pipeline runner: wraps standalone.py execution with live log streaming,
thread management, and override patching.
"""
import json
import logging
import os
import queue
import sys
import threading
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import standalone as _sa

from backend.services.storage import (
    get_prompts, get_config, save_run_result, save_version, get_run_result
)

# ── Fix: non-sequential ID normalisation ─────────────────────────────────────
# The LLM often returns sentence IDs as "S1", "S2" etc. (matching the "S{n}:"
# format shown in the prompt) even though the response schema asks for numbers.
# standalone.py's _assemble_nonseq_short calls int(sid) which throws on "S123",
# silently skips every ID and returns None — producing 0 non-sequential shorts.
# We monkey-patch the function to strip the leading "S" before int-casting.
_orig_assemble = _sa._assemble_nonseq_short

def _patched_assemble(sentence_ids, sentence_by_idx, topic="", reason=""):
    normalised = []
    for sid in sentence_ids:
        if isinstance(sid, str):
            sid = sid.strip().lstrip("Ss")  # "S123" → "123"
        normalised.append(sid)
    return _orig_assemble(normalised, sentence_by_idx, topic=topic, reason=reason)

_sa._assemble_nonseq_short = _patched_assemble


def reload_standalone():
    """Hot-reload standalone.py after a code edit and re-apply monkey-patches."""
    import importlib
    global _sa, _orig_assemble
    importlib.reload(_sa)
    # Capture the freshly-loaded original, then re-apply the patch
    _orig_assemble = _sa._assemble_nonseq_short
    _sa._assemble_nonseq_short = _patched_assemble


# ── Global run state ──────────────────────────────────────────────────────────
_runs: Dict[str, "RunState"] = {}
_active_run_id: Optional[str] = None
_run_lock = threading.Lock()


class RunState:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.status = "starting"
        self.logs: list = []
        self.result: Optional[Dict] = None
        self.error: Optional[str] = None
        self.log_queue: queue.Queue = queue.Queue()
        self.start_time = datetime.utcnow().isoformat()
        self.end_time: Optional[str] = None


# ── Custom log handler for live streaming ─────────────────────────────────────

class _QueueLogHandler(logging.Handler):
    """Captures records from the standalone logger and pushes to a queue."""

    _FMT = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                              datefmt="%H:%M:%S")

    def __init__(self, run_state: RunState):
        super().__init__()
        self.run_state = run_state

    def emit(self, record: logging.LogRecord):
        msg = {
            "type": "log",
            "level": record.levelname,
            "message": self._FMT.format(record),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.run_state.logs.append(msg)
        self.run_state.log_queue.put(msg)


# ── Module patching ───────────────────────────────────────────────────────────

def _apply_overrides(prompts: Dict[str, str], config: Dict[str, Any]):
    """Monkey-patch standalone.py module globals before execution."""
    for key, value in {**prompts, **config}.items():
        if hasattr(_sa, key) and value not in (None,):
            setattr(_sa, key, value)

    # Always reset stateful singletons so a fresh client is created
    _sa._client = None
    _sa._token_usage = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cache_read_tokens": 0,
        "total_cache_write_tokens": 0,
        "calls": [],
    }


# ── Pipeline thread ───────────────────────────────────────────────────────────

def _run_pipeline(
    run_state: RunState,
    transcript_path: str,
    video_path: Optional[str],
    clip_mode: str,
):
    global _active_run_id

    sa_logger = logging.getLogger("clipping_agent_standalone")
    handler = _QueueLogHandler(run_state)
    sa_logger.addHandler(handler)

    def _log(level, msg):
        entry = {"type": "log", "level": level, "message": msg,
                 "timestamp": datetime.utcnow().isoformat()}
        run_state.logs.append(entry)
        run_state.log_queue.put(entry)

    try:
        run_state.status = "running"
        _log("INFO", f"[PLAYGROUND] Run {run_state.run_id} started")
        _log("INFO", f"[PLAYGROUND] Transcript: {transcript_path}")
        _log("INFO", f"[PLAYGROUND] Video: {video_path or 'none'}")
        _log("INFO", f"[PLAYGROUND] Mode: {clip_mode}")

        # Load transcript
        with open(transcript_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        transcription_data = raw.get("results", raw)

        # Run pipeline (blocking)
        result = _sa.run(
            transcription_data,
            video_path=video_path if video_path and os.path.exists(video_path) else None,
            clip_mode=clip_mode,
        )

        run_state.status = "success"
        run_state.result = result
        run_state.end_time = datetime.utcnow().isoformat()

        summary = {
            "sequential_count": len(result.get("sequential_clips", [])),
            "non_sequential_count": len(result.get("non_sequential_shorts", [])),
            "video_duration": result.get("video_duration", 0),
            "category": result.get("video_category", ""),
            "total_tokens": result.get("token_usage", {}).get("total_tokens", 0),
        }

        # Persist to disk
        save_run_result(run_state.run_id, {
            "run_id": run_state.run_id,
            "status": "success",
            "start_time": run_state.start_time,
            "end_time": run_state.end_time,
            "summary": summary,
            "result": result,
        })

        run_state.log_queue.put({
            "type": "done",
            "status": "success",
            "summary": summary,
        })

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        run_state.status = "error"
        run_state.error = err
        run_state.end_time = datetime.utcnow().isoformat()
        _log("ERROR", f"[PIPELINE ERROR] {err}")
        run_state.log_queue.put({"type": "done", "status": "error", "error": str(exc)})

    finally:
        sa_logger.removeHandler(handler)
        with _run_lock:
            _active_run_id = None


# ── Public API ─────────────────────────────────────────────────────────────────

def start_run(
    transcript_path: str,
    video_path: Optional[str],
    clip_mode: str = "both",
    save_version_flag: bool = False,
    prompt_overrides: Optional[Dict[str, str]] = None,
    code_override: Optional[str] = None,
) -> str:
    global _active_run_id

    with _run_lock:
        if _active_run_id:
            raise RuntimeError(f"Run {_active_run_id} is already in progress.")

        run_id = uuid.uuid4().hex[:8]
        run_state = RunState(run_id)
        _runs[run_id] = run_state
        _active_run_id = run_id

    # Use request-time prompt overrides if provided, otherwise fall back to disk values.
    # This keeps the run session-local — nothing is written to disk.
    prompts = prompt_overrides if prompt_overrides is not None else get_prompts()
    config = get_config()
    _apply_overrides(prompts, config)

    # Apply session-only code override via exec — no disk write.
    if code_override:
        try:
            exec(compile(code_override, "<session_code_override>", "exec"), vars(_sa))
        except Exception as e:
            raise RuntimeError(f"Code override failed to compile/execute: {e}")

    if save_version_flag:
        save_version(note=f"Auto-save before run {run_id}")

    thread = threading.Thread(
        target=_run_pipeline,
        args=(run_state, transcript_path, video_path, clip_mode),
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
            "run_id": state.run_id,
            "status": state.status,
            "start_time": state.start_time,
            "end_time": state.end_time,
            "error": state.error,
            "log_count": len(state.logs),
            "has_result": state.result is not None,
        }
    # Try disk
    saved = get_run_result(run_id)
    if saved:
        return {
            "run_id": saved.get("run_id"),
            "status": saved.get("status"),
            "start_time": saved.get("start_time"),
            "end_time": saved.get("end_time"),
            "log_count": 0,
            "has_result": True,
        }
    return None
