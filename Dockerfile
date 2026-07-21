# ============================================================================
# Stage 1: Builder — install Python dependencies into an isolated prefix
# ============================================================================
FROM python:3.12-slim AS builder

# Build-time deps for compiled wheels (sentence-transformers, chromadb, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================================
# Stage 2: Runtime — lean production image
# ============================================================================
FROM python:3.12-slim AS runtime

# APP_ENV=production makes the image fail-closed: app/auth.py refuses to start
# with the built-in JWT_SECRET default, and database.py refuses to seed the
# admin account without an explicit ADMIN_PASSWORD. docker-compose overrides
# this to "development" for local runs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    HF_HOME=/app/data/huggingface

# curl — HEALTHCHECK probe; git — required by gitpython (loaders.py)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-built site-packages from the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY frontend/ ./frontend/

# Non-root user for runtime security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --no-create-home --shell /bin/false appuser && \
    mkdir -p /app/chroma_db /app/data/huggingface && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# PORT is injected by Render/Railway; falls back to 8000 for local/Azure runs
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
