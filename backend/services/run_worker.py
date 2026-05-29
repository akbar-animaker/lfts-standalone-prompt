"""
Per-run worker — spawned as a subprocess by runner.py.
Each run gets its own Python process and completely isolated standalone.py state.
Reads all args from a JSON file, runs the pipeline, streams log+done messages
to stdout as newline-delimited JSON.
"""
import json
import logging
import os
import sys
import tempfile
import traceback


class _StdoutJsonHandler(logging.Handler):
    _FMT = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    def emit(self, record):
        _out({"type": "log", "level": record.levelname, "message": self._FMT.format(record)})


def _out(obj):
    print(json.dumps(obj), flush=True)


def main(args_file: str):
    with open(args_file, "r", encoding="utf-8") as f:
        args = json.load(f)

    run_id       = args["run_id"]
    transcript   = args["transcript_path"]
    video_path   = args.get("video_path") or ""
    clip_mode    = args.get("clip_mode", "both")
    config       = args.get("config", {})
    prompt_ovr   = args.get("prompt_overrides") or {}
    code_ovr     = args.get("code_override")
    output_dir   = args["output_dir"]

    # Wire standalone logger → stdout JSON
    sa_logger = logging.getLogger("clipping_agent_standalone")
    sa_logger.addHandler(_StdoutJsonHandler())
    sa_logger.setLevel(logging.DEBUG)

    def _log(level, msg):
        _out({"type": "log", "level": level, "message": msg})

    try:
        # Make standalone.py importable from any working directory
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, root)

        import standalone as _sa  # isolated import — this process only

        # ── Apply config snapshot (model, API keys, durations, etc.) ─────────
        for k, v in config.items():
            if hasattr(_sa, k) and v is not None:
                setattr(_sa, k, v)

        # ── Apply prompt overrides on top (session drafts win over config) ───
        for k, v in prompt_ovr.items():
            if hasattr(_sa, k) and v is not None:
                setattr(_sa, k, v)

        # ── Redirect output files to this run's isolated directory ────────────
        os.makedirs(output_dir, exist_ok=True)
        _sa.OUTPUT_PATH = os.path.join(output_dir, "result.json")
        if hasattr(_sa, "SPEAKERS_OUTPUT_PATH"):
            _sa.SPEAKERS_OUTPUT_PATH = os.path.join(output_dir, "speakers.json")

        # ── Reset per-run singletons ──────────────────────────────────────────
        _sa._client = None
        _sa._token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_write_tokens": 0,
            "calls": [],
        }

        # ── Monkey-patch: normalise "S123" → 123 for non-sequential IDs ──────
        _orig = _sa._assemble_nonseq_short
        def _patched(sids, sentence_by_idx, topic="", reason=""):
            norm = []
            for sid in sids:
                if isinstance(sid, str):
                    sid = sid.strip().lstrip("Ss")
                norm.append(sid)
            return _orig(norm, sentence_by_idx, topic=topic, reason=reason)
        _sa._assemble_nonseq_short = _patched

        # ── Apply session-only code override ──────────────────────────────────
        if code_ovr:
            exec(compile(code_ovr, "<session_code_override>", "exec"), vars(_sa))

        # ── Load transcript (handle URLs) ─────────────────────────────────────
        _tmp = None
        local_t = transcript
        if transcript.startswith(("http://", "https://")):
            _log("INFO", "[WORKER] Transcript is a URL — downloading…")
            import urllib.request
            _tmp = tempfile.mktemp(suffix=".json")
            urllib.request.urlretrieve(transcript, _tmp)
            local_t = _tmp
        try:
            with open(local_t, "r", encoding="utf-8") as f:
                raw = json.load(f)
        finally:
            if _tmp and os.path.exists(_tmp):
                os.unlink(_tmp)

        transcription_data = raw.get("results", raw)

        _log("INFO", f"[WORKER] Run {run_id} started — mode={clip_mode}")
        _log("INFO", f"[WORKER] Transcript: {transcript}")
        _log("INFO", f"[WORKER] Video: {video_path or 'none'}")
        _log("INFO", f"[WORKER] Output dir: {output_dir}")

        # ── Execute pipeline ──────────────────────────────────────────────────
        result = _sa.run(
            transcription_data,
            video_path=video_path if video_path and os.path.exists(video_path) else None,
            clip_mode=clip_mode,
        )

        _out({"type": "done", "status": "success", "result": result})

    except Exception as exc:
        tb = traceback.format_exc()
        _log("ERROR", f"[WORKER ERROR] {type(exc).__name__}: {exc}\n{tb}")
        _out({"type": "done", "status": "error", "error": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_worker.py <args_file>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
