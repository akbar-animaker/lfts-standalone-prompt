"""
Resolve a project ID via the transcribe API, then download the transcript and
video to local storage so the existing pipeline can run on them.

API:  POST https://snbx.vmakerdev.com/lstsf/transcribe/   body: {"projectId": "<id>"}
Success → {"status":"success","data":{"video_url":..., "transcribe_url":...}}
Failure → {"status":"error","message":"..."}
"""
import json
import logging
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request

# boto3 is optional — if unavailable we just download over HTTPS.
try:
    import boto3
except ImportError:
    boto3 = None

log = logging.getLogger("clipping_agent_standalone")

API_BASE = "https://snbx.vmakerdev.com/lstsf/transcribe"

# When the API hands back a dash.animaker.com URL, the same object lives in S3.
# Pulling it directly from the bucket is far faster than the web/proxy layer.
S3_SOURCE_HOST = "dash.animaker.com"
S3_BUCKET = "anim-user-uploads"

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOWNLOADS_DIR = os.path.join(_BASE, "storage", "downloads")

_HTTP_TIMEOUT = 30      # seconds — API metadata call
_DL_TIMEOUT = 600       # seconds — file downloads (video can be large)


def _safe_id(pid: str) -> str:
    """Filesystem-safe directory name for a project."""
    cleaned = "".join(c for c in pid if c.isalnum() or c in "-_")
    return cleaned or "project"


def _http_post_json(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, dest: str):
    req = urllib.request.Request(url, headers={"User-Agent": "lfts-playground"})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=_DL_TIMEOUT) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    os.replace(tmp, dest)


def _download_via_s3(key: str, dest: str):
    """Download s3://S3_BUCKET/<key> using boto3 session authentication."""
    session = boto3.Session()                 # default credential chain (env / profile / IAM role)
    s3 = session.client("s3")
    tmp = dest + ".part"
    try:
        s3.download_file(S3_BUCKET, key, tmp)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _download_smart(url: str, dest: str):
    """Prefer S3 for dash.animaker.com URLs (much faster); fall back to HTTPS.

    The S3 object key is the URL path with the leading slash removed, e.g.
    https://dash.animaker.com/a/u/x/video/proxy1/86683kg.mp4
        → key  a/u/x/video/proxy1/86683kg.mp4
    """
    parsed = urllib.parse.urlparse(url)
    key = parsed.path.lstrip("/")
    if boto3 and parsed.hostname == S3_SOURCE_HOST and key:
        try:
            log.info(f"[PROJECT] Downloading via S3: s3://{S3_BUCKET}/{key}")
            _download_via_s3(key, dest)
            return
        except Exception as e:
            log.warning(f"[PROJECT] S3 download failed ({e}); falling back to HTTPS")
    _download(url, dest)


def fetch_project(project_id: str) -> dict:
    """Resolve a project's URLs and download both files locally.

    Returns {project_id, transcript_path, video_path, video_url, transcribe_url, warning}.
    Raises RuntimeError with a user-facing message on any hard failure
    (missing ID, API error, or transcript download failure). A failed *video*
    download is soft — it returns an empty video_path plus a `warning`.
    """
    project_id = (project_id or "").strip()
    if not project_id:
        raise RuntimeError("Project ID is required")

    api_url = f"{API_BASE}/"
    log.info(f"[PROJECT] Resolving '{project_id}' via POST {api_url}")
    try:
        payload = _http_post_json(api_url, {"projectId": project_id})
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Transcribe API returned HTTP {e.code} for project '{project_id}'")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach the transcribe API: {e.reason}")
    except (json.JSONDecodeError, ValueError):
        raise RuntimeError("Transcribe API returned an unexpected (non-JSON) response")

    if payload.get("status") != "success":
        raise RuntimeError(payload.get("message") or "Transcribe API returned an error")

    data = payload.get("data") or {}
    transcribe_url = data.get("transcribe_url")
    video_url = data.get("video_url")
    if not transcribe_url:
        raise RuntimeError("API response did not include a transcribe_url")

    proj_dir = os.path.join(DOWNLOADS_DIR, _safe_id(project_id))
    os.makedirs(proj_dir, exist_ok=True)

    transcript_path = os.path.join(proj_dir, "transcript.json")
    log.info("[PROJECT] Downloading transcript...")
    try:
        _download_smart(transcribe_url, transcript_path)
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"Failed to download transcript: {e}")

    video_path = ""
    warning = ""
    if video_url:
        ext = os.path.splitext(urllib.parse.urlparse(video_url).path)[1] or ".mp4"
        candidate = os.path.join(proj_dir, f"video{ext}")
        log.info("[PROJECT] Downloading video...")
        try:
            _download_smart(video_url, candidate)
            video_path = candidate
        except (urllib.error.URLError, OSError) as e:
            warning = f"Video download failed ({e}) — running without video (multimodal disabled)."
            log.warning(f"[PROJECT] {warning}")
    else:
        warning = "No video_url in API response — running transcript-only."

    log.info(f"[PROJECT] Ready: transcript={transcript_path} video={video_path or 'none'}")
    return {
        "project_id": project_id,
        "transcript_path": transcript_path,
        "video_path": video_path,
        "video_url": video_url,
        "transcribe_url": transcribe_url,
        "warning": warning,
    }
