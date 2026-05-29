"""
All API routes for the Prompt Engineering Playground.
"""
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.services import storage, runner, projects

router = APIRouter(prefix="/api")


# ── Prompts ────────────────────────────────────────────────────────────────────

@router.get("/prompts")
def get_prompts():
    return storage.get_prompts()


@router.put("/prompts")
def update_prompts(updates: Dict[str, str]):
    return storage.save_prompts(updates)


@router.post("/prompts/{name}/reset")
def reset_prompt(name: str):
    default_val = storage.reset_prompt(name)
    return {"name": name, "value": default_val}


# ── Config ─────────────────────────────────────────────────────────────────────

@router.get("/config")
def get_config():
    cfg = storage.get_config()
    # Mask API keys for display
    for key_name in ("CLAUDE_API_KEY", "OPENAI_API_KEY"):
        key = cfg.get(key_name) or ""
        if len(key) > 12:
            cfg[key_name] = key[:8] + "..." + key[-4:]
        elif key:
            cfg[key_name] = "***"
    return cfg


@router.get("/config/schema")
def get_config_schema():
    return storage.get_config_schema()


@router.put("/config")
def update_config(updates: Dict[str, Any]):
    return storage.save_config(updates)


@router.post("/config/reset")
def reset_config():
    return storage.reset_config()


# ── Execution ──────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    clip_mode: str = "both"
    transcript_path: Optional[str] = None
    video_path: Optional[str] = None
    save_version: bool = False
    prompt_overrides: Optional[Dict[str, str]] = None   # session-only, never written to disk
    code_override: Optional[str] = None                 # session-only, applied via exec


@router.post("/run")
def start_run(req: RunRequest):
    import standalone as _sa

    # Read transcript/video from config if not provided in request
    transcript_path = req.transcript_path or _sa.TRANSCRIPT_PATH
    video_path = req.video_path if req.video_path is not None else _sa.VIDEO_PATH

    # Allow URL paths — they are resolved inside the worker subprocess
    _is_url = transcript_path.startswith(("http://", "https://"))
    if not _is_url and not os.path.exists(transcript_path):
        raise HTTPException(
            status_code=400,
            detail=f"Transcript not found: {transcript_path}"
        )

    run_id = runner.start_run(
        transcript_path=transcript_path,
        video_path=video_path,
        clip_mode=req.clip_mode,
        save_version_flag=req.save_version,
        prompt_overrides=req.prompt_overrides,
        code_override=req.code_override,
    )
    return {"run_id": run_id, "status": "started"}


@router.get("/stream/{run_id}")
async def stream_logs(run_id: str):
    """Server-Sent Events endpoint for live log streaming."""
    state = runner.get_run(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        log_q = state.log_queue
        loop = asyncio.get_event_loop()

        # Replay already-captured logs so late-connecting clients get full history
        for entry in list(state.logs):
            yield f"data: {json.dumps(entry)}\n\n"

        # Stream new entries
        while True:
            try:
                msg = await loop.run_in_executor(
                    None, lambda: log_q.get(timeout=1.0)
                )
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") == "done":
                    break
            except Exception:
                # Timeout or queue error
                if state.status in ("success", "error"):
                    yield f"data: {json.dumps({'type':'done','status':state.status})}\n\n"
                    break
                # keepalive comment so the connection stays open
                yield ": ping\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/runs")
def list_runs():
    """List all saved run results, newest first."""
    return {"runs": storage.list_runs()}


@router.get("/runs/active")
def get_active_run():
    active_id = runner.get_active_run_id()
    return {"active_run_id": active_id}


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    status = runner.get_run_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="Run not found")
    return status


@router.get("/runs/{run_id}/result")
def get_run_result(run_id: str):
    """Return the full result for a completed run."""
    state = runner.get_run(run_id)
    if state and state.result:
        return state.result

    saved = storage.get_run_result(run_id)
    if saved:
        return saved.get("result") or saved
    raise HTTPException(status_code=404, detail="Result not found")


# ── Versions ───────────────────────────────────────────────────────────────────

@router.get("/versions")
def get_versions():
    return {"versions": storage.get_versions()}


class SaveVersionRequest(BaseModel):
    note: str = ""


@router.post("/versions")
def save_version(req: SaveVersionRequest):
    return storage.save_version(note=req.note)


@router.get("/versions/{version_id}")
def get_version(version_id: str):
    v = storage.get_version(version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    return v


@router.delete("/versions/{version_id}")
def delete_version(version_id: str):
    storage.delete_version(version_id)
    return {"success": True}


@router.post("/versions/{version_id}/restore")
def restore_version(version_id: str):
    ok = storage.restore_version(version_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"success": True, "message": "Prompts and config restored"}


# ── Code Editor ───────────────────────────────────────────────────────────────

_CODE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "standalone.py",
)


@router.get("/code")
def get_code():
    try:
        with open(_CODE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content, "path": _CODE_PATH}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="standalone.py not found")


class CodeUpdateRequest(BaseModel):
    content: str


@router.put("/code")
def update_code(req: CodeUpdateRequest):
    """Save standalone.py then hot-reload the module so the next run uses the new code."""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    with open(_CODE_PATH, "w", encoding="utf-8") as f:
        f.write(req.content)
    warn = None
    try:
        runner.reload_standalone()
    except Exception as e:
        warn = str(e)
    return {"success": True, "reload_warning": warn}


# ── Project (fetch transcript + video by ID) ─────────────────────────────────

class ProjectFetchRequest(BaseModel):
    project_id: str


@router.post("/project/fetch")
def project_fetch(req: ProjectFetchRequest):
    """Resolve a project ID via the transcribe API, download the transcript +
    video locally, and point the pipeline (and UI) at the downloaded files."""
    import standalone as _sa
    try:
        result = projects.fetch_project(req.project_id)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Store URLs in config (for display) but point the live module at local paths
    # so the current session can serve the video and run immediately.
    _sa.TRANSCRIPT_PATH = result["transcript_path"]
    _sa.VIDEO_PATH = result["video_path"] or ""
    storage.save_config({
        "TRANSCRIPT_PATH": result["transcribe_url"] or result["transcript_path"],
        "VIDEO_PATH": result["video_url"] or result["video_path"] or "",
    })
    return {"status": "success", **result}


# ── Video ──────────────────────────────────────────────────────────────────────

@router.get("/video")
async def serve_video(request: Request):
    """
    Serves the video file with HTTP Range support so the HTML5 <video>
    element can seek to any timestamp (required for clip preview).
    """
    import standalone as _sa
    from fastapi.responses import RedirectResponse
    video_path = _sa.VIDEO_PATH
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found. Check VIDEO_PATH in Config.")
    # If it's a URL, redirect the browser directly to it
    if video_path.startswith(("http://", "https://")):
        return RedirectResponse(url=video_path)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found. Check VIDEO_PATH in Config.")

    file_size = os.path.getsize(video_path)
    range_header = request.headers.get("range")

    # Detect MIME type from extension
    ext = os.path.splitext(video_path)[1].lower()
    mime = {"mp4": "video/mp4", "mov": "video/quicktime",
            "avi": "video/x-msvideo", "mkv": "video/x-matroska"}.get(ext.lstrip("."), "video/mp4")

    if not range_header:
        # Full-file response (needed for initial metadata load)
        return StreamingResponse(
            _iter_file(video_path, 0, file_size - 1),
            status_code=200,
            media_type=mime,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )

    # Parse "bytes=start-end"
    m = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not m:
        raise HTTPException(status_code=416, detail="Malformed Range header")

    start = int(m.group(1))
    # Default chunk: 2 MB so seeking is snappy
    end = int(m.group(2)) if m.group(2) else min(start + 2 * 1024 * 1024, file_size - 1)
    end = min(end, file_size - 1)

    if start > end or start >= file_size:
        raise HTTPException(status_code=416, detail="Range Not Satisfiable")

    length = end - start + 1
    return StreamingResponse(
        _iter_file(video_path, start, end),
        status_code=206,
        media_type=mime,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )


def _iter_file(path: str, start: int, end: int, chunk: int = 65536):
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            data = f.read(min(chunk, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


@router.get("/video/info")
def video_info():
    """Returns metadata about the video file for the player UI."""
    import standalone as _sa
    vp = _sa.VIDEO_PATH
    is_url = bool(vp and vp.startswith(("http://", "https://")))
    exists = bool(vp and (is_url or os.path.exists(vp)))
    return {
        "exists": exists,
        "path": vp,
        "size_mb": round(os.path.getsize(vp) / 1024 / 1024, 1) if (exists and not is_url) else None,
        "url": "/api/video" if exists else None,
    }


# ── Storage management ────────────────────────────────────────────────────────

@router.get("/storage/info")
def get_storage_info():
    return storage.get_storage_info()


class ClearStorageRequest(BaseModel):
    areas: list


@router.post("/storage/clear")
def clear_storage(req: ClearStorageRequest):
    result = storage.clear_storage_areas(req.areas)
    return result


# ── Export ─────────────────────────────────────────────────────────────────────

@router.get("/export/current")
def export_current(format: str = "json"):
    """Export current prompts or config (no run required)."""
    if format == "prompts":
        content = json.dumps(storage.get_prompts(), indent=2, ensure_ascii=False)
        return StreamingResponse(iter([content]), media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=prompts.json"})
    elif format == "config":
        content = json.dumps(storage.get_config(), indent=2, ensure_ascii=False)
        return StreamingResponse(iter([content]), media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=config.json"})
    raise HTTPException(status_code=400, detail="Use /export/{run_id} for run results")


@router.get("/export/{run_id}")
def export_run(run_id: str, format: str = "json"):
    state = runner.get_run(run_id)
    if state and state.result:
        result = state.result
    else:
        saved = storage.get_run_result(run_id)
        if saved:
            result = saved.get("result") or saved
        else:
            raise HTTPException(status_code=404, detail="Run not found")

    if format == "json":
        content = json.dumps(result, indent=2, ensure_ascii=False)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=result_{run_id}.json"},
        )
    elif format == "markdown":
        md = _to_markdown(result)
        return StreamingResponse(
            iter([md]),
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=result_{run_id}.md"},
        )
    elif format == "prompts":
        prompts = storage.get_prompts()
        content = json.dumps(prompts, indent=2, ensure_ascii=False)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=prompts.json"},
        )
    elif format == "config":
        config = storage.get_config()
        content = json.dumps(config, indent=2, ensure_ascii=False)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=config.json"},
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown format: {format}")


def _to_markdown(result: Dict) -> str:
    lines = [
        "# Pipeline Result\n",
        f"- **Category**: {result.get('video_category', 'unknown')}",
        f"- **Duration**: {result.get('video_duration', 0):.1f}s",
        f"- **Language**: {result.get('language_code', 'unknown')}",
        f"- **Sequential Clips**: {len(result.get('sequential_clips', []))}",
        f"- **Non-Sequential Shorts**: {len(result.get('non_sequential_shorts', []))}",
        "",
    ]
    lines.append("## Category Reasoning\n")
    lines.append(result.get("category_reason", "") + "\n")

    lines.append("## Sequential Clips\n")
    for i, clip in enumerate(result.get("sequential_clips", []), 1):
        dur = clip.get("video_end_time", 0) - clip.get("video_start_time", 0)
        lines += [
            f"### {i}. {clip.get('title', 'Untitled')}",
            f"- **Score**: {clip.get('confidence_score', 0)}/100",
            f"- **Duration**: {dur:.1f}s  ({clip.get('video_start_time',0):.1f}s – {clip.get('video_end_time',0):.1f}s)",
            f"- **Platforms**: {', '.join(clip.get('social_media', []))}",
            f"- **Reason**: {clip.get('reason', '')}",
            "",
        ]

    lines.append("## Non-Sequential Shorts\n")
    for i, short in enumerate(result.get("non_sequential_shorts", []), 1):
        lines += [
            f"### {i}. {short.get('title', 'Untitled')}",
            f"- **Score**: {short.get('confidence_score', 0)}/100",
            f"- **Duration**: {short.get('total_duration', 0):.1f}s",
            f"- **Segments**: {short.get('num_clips', 0)}",
            f"- **Platforms**: {', '.join(short.get('social_media', []))}",
            "",
        ]

    usage = result.get("token_usage", {})
    if usage:
        lines += [
            "## Token Usage\n",
            f"- Total: {usage.get('total_tokens', 0):,}",
            f"- Input: {usage.get('total_input_tokens', 0):,}",
            f"- Output: {usage.get('total_output_tokens', 0):,}",
            f"- Cache Read: {usage.get('total_cache_read_tokens', 0):,}",
            f"- LLM Calls: {usage.get('llm_calls', 0)}",
        ]

    return "\n".join(lines)


# ── Status ─────────────────────────────────────────────────────────────────────

@router.get("/status")
def get_status():
    import standalone as _sa
    return {
        "active_run_id": runner.get_active_run_id(),
        "transcript_exists": os.path.exists(_sa.TRANSCRIPT_PATH),
        "video_exists": os.path.exists(_sa.VIDEO_PATH or ""),
        "transcript_path": _sa.TRANSCRIPT_PATH,
        "video_path": _sa.VIDEO_PATH,
        "model": _sa.CLAUDE_MODEL,
        "clip_mode": _sa.CLIP_MODE,
    }
