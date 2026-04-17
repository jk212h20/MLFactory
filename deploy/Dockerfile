# Dockerfile for the MLFactory AZ service (FastAPI).
#
# Serves the AlphaZero network over HTTP. Linked into the Boop game site
# via AZ_SERVICE_URL. See service/app.py for endpoints.
#
# Design notes:
# - Use python:3.13-slim: matches dev, small, current torch wheels available.
# - Install uv for fast locked deps (uv.lock is the source of truth).
# - Torch CPU-only: we're serving batch-1 inference; MPS/CUDA not needed.
# - The checkpoint is baked into the image at /app/checkpoints/; the
#   AZ_CHECKPOINT env var (set in Railway) points at it. Small (~1.3 MB).
# - HEALTHCHECK hits /health so Railway knows when the service is ready.

FROM python:3.13-slim

# System deps: git (for anything uv may need) and curl (for healthcheck).
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install uv from its official installer (pinned indirectly via image digest).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy the project files first (for layer caching of deps).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install into the system python environment (no venv) using the locked file.
# --frozen keeps it deterministic. --no-dev skips test-only deps.
RUN uv pip install --system --no-cache --python $(which python3) \
    "torch==2.11.0" "numpy==2.3.4" "rich>=14.2.0" "typer>=0.20.0" \
    "pyyaml>=6.0.3" "fastapi>=0.120.2" "uvicorn[standard]>=0.44.0" \
 && uv pip install --system --no-cache --no-deps --python $(which python3) .

# Ship the checkpoint in the image.
COPY deploy/checkpoints /app/checkpoints

ENV AZ_CHECKPOINT=/app/checkpoints/boop-run52-iter15.pt \
    AZ_DEFAULT_SIMS=200 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://127.0.0.1:${PORT:-8000}/health || exit 1

# Railway sets $PORT; fall back to 8000 locally. `--host 0.0.0.0` so the
# service is reachable on the container's interface.
CMD ["sh", "-c", "uvicorn mlfactory.service.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
