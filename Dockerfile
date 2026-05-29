# ── Stage 1: builder — install Python deps ───────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy both requirement files and install everything in one layer
COPY requirements.txt playground_requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install \
        -r requirements.txt \
        -r playground_requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="Prompt Engineering Playground" \
      org.opencontainers.image.description="Interactive UI for the Video Clipping Pipeline"

# ffmpeg is needed by standalone.py for video frame extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# App working directory
WORKDIR /app

# Copy application source (see .dockerignore for exclusions)
COPY backend/       ./backend/
COPY frontend/      ./frontend/
COPY standalone.py  ./
COPY playground.py  ./

# Persistent data lives in a volume so it survives container restarts:
#   /app/data/       — transcript.json, input.mp4 (user-supplied media)
#   /app/storage/    — saved prompts, configs, versions, run results
RUN mkdir -p /app/data /app/storage

# Runtime configuration — all overridable via `docker run -e` or docker-compose env
ENV PLAYGROUND_HOST=0.0.0.0 \
    PLAYGROUND_PORT=8765 \
    CLAUDE_API_KEY="" \
    TRANSCRIPT_PATH=/app/data/transcript.json \
    VIDEO_PATH=/app/data/input.mp4

EXPOSE 8765

# Health check — verifies the API is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/api/status')" || exit 1

# Disable the automatic browser-open (no display in a container)
# and launch uvicorn directly so signals are handled cleanly
CMD ["python", "-u", "playground.py"]
