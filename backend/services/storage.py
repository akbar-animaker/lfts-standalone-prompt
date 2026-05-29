"""
Persistent storage for prompts, config, versions, and run results.
Uses JSON files in the storage/ directory.
"""
import json
import os
import shutil
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import standalone to read default values
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import standalone as _sa

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORAGE_DIR = os.path.join(_BASE, "storage")
PROMPTS_FILE = os.path.join(STORAGE_DIR, "prompts.json")
CONFIG_FILE  = os.path.join(STORAGE_DIR, "config.json")
VERSIONS_FILE = os.path.join(STORAGE_DIR, "versions.json")
RUNS_DIR = os.path.join(STORAGE_DIR, "runs")

_lock = threading.Lock()

# ── Prompt names exposed in the UI ────────────────────────────────────────────
PROMPT_NAMES = [
    "SEQUENCE_PROMPT",
    "REFINE_PROMPT",
    "NONSEQUENCE_PROMPT",
    "REFINE_NONSEQ_PROMPT",
    "SCORER_PROMPT",
    "METADATA_PROMPT",
    "CATEGORY_DETECTION_PROMPT",
    "SPEAKER_DETECTION_PROMPT",
    "THUMBNAIL_ART_DIRECTOR_SYSTEM",
    "THUMBNAIL_IMAGE_INSTRUCTION",
]

# ── Config keys exposed in the UI (key → (type, label, group)) ────────────────
CONFIG_SCHEMA = {
    "CLAUDE_MODEL":                 (str,   "Model Name",               "Model"),
    "CLAUDE_MAX_TOKENS":            (int,   "Max Tokens",               "Model"),
    "CLAUDE_API_KEY":               (str,   "API Key",                  "Model"),
    "CLIP_MODE":                    (str,   "Clip Mode",                "Pipeline"),
    "SEQUENTIAL_MIN_DURATION":      (int,   "Sequential Min Duration (s)","Pipeline"),
    "SOCIAL_MEDIA_MAX_DURATION":    (int,   "Social Media Max Duration (s)","Pipeline"),
    "NON_SEQUENTIAL_MIN_DURATION":  (int,   "Non-Sequential Min Duration (s)","Pipeline"),
    "NON_SEQUENTIAL_MAX_DURATION":  (int,   "Non-Sequential Max Duration (s)","Pipeline"),
    "SECTION_BREAK_GAP_THRESHOLD":  (float, "Section Break Gap (s)",    "Pipeline"),
    "TARGET_SCORE":                 (int,   "Target Score (0-100)",     "Scoring"),
    "MAX_ITERATIONS_PER_SHORT":     (int,   "Max Refinement Iterations","Scoring"),
    "SCORE_MARKET_ADJUSTMENT":      (int,   "Score Market Adjustment",  "Scoring"),
    "REFINE_PAD":                   (float, "Refine Pad (s)",           "Refinement"),
    "REFINE_NONSEQ_PAD":            (float, "Non-Seq Refine Pad (s)",   "Refinement"),
    "FRAME_INTERVAL":               (int,   "Frame Interval (s)",       "Video"),
    "MAX_FRAMES":                   (int,   "Max Frames",               "Video"),
    "TRANSCRIPT_PATH":              (str,   "Transcript URL",           "Files"),
    "VIDEO_PATH":                   (str,   "Video URL",                "Files"),
    "OUTPUT_PATH":                  (str,   "Output Path",              "Files"),
    "ENABLE_THUMBNAILS":            (bool,  "Enable Thumbnails",        "Thumbnails"),
    "OPENAI_API_KEY":               (str,   "OpenAI API Key",           "Thumbnails"),
    "THUMBNAIL_RESPONSE_MODEL":     (str,   "Thumbnail Model",          "Thumbnails"),
    "MAX_THUMBNAILS_PER_TYPE":      (int,   "Max Thumbnails Per Type",  "Thumbnails"),
    "S3_BUCKET":                    (str,   "S3 Bucket",                "Thumbnails"),
}


def _cast_bool(v: Any) -> bool:
    """Coerce UI values ("true"/"false", 0/1, bool) into a real bool."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ensure_dirs():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)


def _read_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Prompts ────────────────────────────────────────────────────────────────────

def get_prompts() -> Dict[str, str]:
    """Return all prompts. Stored overrides take priority, else module defaults."""
    _ensure_dirs()
    stored = _read_json(PROMPTS_FILE, {})
    result = {}
    for name in PROMPT_NAMES:
        result[name] = stored[name] if name in stored else getattr(_sa, name, "")
    return result


def save_prompts(updates: Dict[str, str]) -> Dict[str, str]:
    with _lock:
        _ensure_dirs()
        current = get_prompts()
        current.update({k: v for k, v in updates.items() if k in PROMPT_NAMES})
        _write_json(PROMPTS_FILE, current)
    return current


def reset_prompt(name: str) -> str:
    """Reset a single prompt to the standalone.py default."""
    with _lock:
        _ensure_dirs()
        current = _read_json(PROMPTS_FILE, {})
        current.pop(name, None)
        _write_json(PROMPTS_FILE, current)
    return getattr(_sa, name, "")


# ── Config ─────────────────────────────────────────────────────────────────────

def get_config() -> Dict[str, Any]:
    """Return all config values. Stored overrides take priority."""
    _ensure_dirs()
    stored = _read_json(CONFIG_FILE, {})
    result = {}
    for key in CONFIG_SCHEMA:
        result[key] = stored[key] if key in stored else getattr(_sa, key, None)
    return result


def get_config_schema() -> Dict:
    """Return metadata about config fields for the UI."""
    schema = {}
    for key, (typ, label, group) in CONFIG_SCHEMA.items():
        schema[key] = {"type": typ.__name__, "label": label, "group": group}
    return schema


def save_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    with _lock:
        _ensure_dirs()
        current = get_config()
        for k, v in updates.items():
            if k in CONFIG_SCHEMA:
                cast_fn = CONFIG_SCHEMA[k][0]
                try:
                    if cast_fn is bool:
                        current[k] = _cast_bool(v)
                    else:
                        current[k] = cast_fn(v) if v not in (None, "") else v
                except (ValueError, TypeError):
                    current[k] = v
        _write_json(CONFIG_FILE, current)
    return current


def reset_config() -> Dict[str, Any]:
    """Reset all config to standalone.py defaults."""
    with _lock:
        _ensure_dirs()
        _write_json(CONFIG_FILE, {})
    return get_config()


# ── Versions ───────────────────────────────────────────────────────────────────

def get_versions() -> List[Dict]:
    _ensure_dirs()
    return _read_json(VERSIONS_FILE, [])


def save_version(note: str = "", result_summary: Optional[Dict] = None) -> Dict:
    with _lock:
        _ensure_dirs()
        versions = get_versions()
        version = {
            "id": datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"),
            "timestamp": datetime.utcnow().isoformat(),
            "note": note,
            "prompts": get_prompts(),
            "config": get_config(),
            "result_summary": result_summary or {},
        }
        versions.insert(0, version)
        # Keep max 50 versions
        if len(versions) > 50:
            versions = versions[:50]
        _write_json(VERSIONS_FILE, versions)
    return version


def get_version(version_id: str) -> Optional[Dict]:
    return next((v for v in get_versions() if v["id"] == version_id), None)


def delete_version(version_id: str) -> bool:
    with _lock:
        versions = [v for v in get_versions() if v["id"] != version_id]
        _write_json(VERSIONS_FILE, versions)
    return True


def restore_version(version_id: str) -> bool:
    version = get_version(version_id)
    if not version:
        return False
    save_prompts(version.get("prompts", {}))
    save_config(version.get("config", {}))
    return True


# ── Run Results ────────────────────────────────────────────────────────────────

def save_run_result(run_id: str, data: Dict):
    _ensure_dirs()
    _write_json(os.path.join(RUNS_DIR, f"{run_id}.json"), data)


def get_run_result(run_id: str) -> Optional[Dict]:
    return _read_json(os.path.join(RUNS_DIR, f"{run_id}.json"), None)


def list_runs() -> List[Dict]:
    _ensure_dirs()
    runs = []
    for fname in sorted(os.listdir(RUNS_DIR), reverse=True):
        if fname.endswith(".json"):
            data = _read_json(os.path.join(RUNS_DIR, fname), {})
            if data:
                runs.append({
                    "run_id": data.get("run_id"),
                    "status": data.get("status"),
                    "start_time": data.get("start_time"),
                    "end_time": data.get("end_time"),
                    "summary": data.get("summary", {}),
                })
    return runs[:20]


# ── Storage management ─────────────────────────────────────────────────────────

_DOWNLOADS_DIR = os.path.join(STORAGE_DIR, "downloads")

STORAGE_AREAS = {
    "runs": {
        "label": "Run Results",
        "description": "One folder per run — result.json, speakers.json, run_args.json",
        "icon": "📊",
        "path": RUNS_DIR,
        "type": "dir",
    },
    "downloads": {
        "label": "Downloaded Projects",
        "description": "Transcripts and videos fetched by Project ID (can be large)",
        "icon": "⬇",
        "path": _DOWNLOADS_DIR,
        "type": "dir",
    },
    "versions": {
        "label": "Version History",
        "description": "Saved prompt + config snapshots",
        "icon": "🕐",
        "path": VERSIONS_FILE,
        "type": "file",
    },
}


def _dir_stats(path: str) -> Dict:
    """Recursively count files and total size (bytes) under path."""
    if not os.path.exists(path):
        return {"files": 0, "size_bytes": 0}
    total_files, total_bytes = 0, 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total_bytes += os.path.getsize(os.path.join(root, f))
                total_files += 1
            except OSError:
                pass
    return {"files": total_files, "size_bytes": total_bytes}


def get_storage_info() -> Dict:
    result = {}
    for key, area in STORAGE_AREAS.items():
        info = {
            "label":       area["label"],
            "description": area["description"],
            "icon":        area["icon"],
        }
        path = area["path"]
        if area["type"] == "dir":
            stats = _dir_stats(path)
            # top-level item count (subdirs or files depending on area)
            if os.path.exists(path):
                items = len(os.listdir(path))
            else:
                items = 0
            info["items"]      = items
            info["files"]      = stats["files"]
            info["size_mb"]    = round(stats["size_bytes"] / 1024 / 1024, 2)
        else:  # single json file
            if os.path.exists(path):
                size = os.path.getsize(path)
                # for versions, items = number of saved versions
                data = _read_json(path, [])
                info["items"]   = len(data) if isinstance(data, list) else 1
                info["files"]   = 1
                info["size_mb"] = round(size / 1024 / 1024, 4)
            else:
                info["items"]   = 0
                info["files"]   = 0
                info["size_mb"] = 0
        result[key] = info
    return result


def clear_storage_areas(areas: List[str]) -> Dict:
    cleared = {}
    for key in areas:
        area = STORAGE_AREAS.get(key)
        if not area:
            continue
        path = area["path"]
        if area["type"] == "dir":
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        else:
            # Reset file to empty list/object
            with _lock:
                _write_json(path, [])
        cleared[key] = True
    return {"cleared": cleared}
