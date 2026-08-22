# ============================================================================
# Stage 1: Builder — install Python dependencies into an isolated prefix
# ============================================================================
FROM python:3.12-slim AS builder

# Build-time deps for compiled wheels (sentence-transformers, chromadb, etc.).
# These never reach the runtime image — that is the point of the split.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CPU-only torch FIRST, so the resolver never pulls the default CUDA build.
# sentence-transformers depends on torch, and the default wheel ships ~1.5 GB
# of NVIDIA libraries that are dead weight on App Service / Render / any CPU
# host. Installing it up front means the requirements.txt pass sees torch as
# already satisfied.
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.12.0

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the embedding + reranking models into the image.
# WHY at build time? Otherwise ~180 MB is fetched from HuggingFace on first use
# — which, in the container, is at *boot*, as a non-root user, over the network.
# A slow or unreachable HF made the pre-warm fail and then took the first real
# request with it. Baking them in makes startup deterministic and offline-safe.
ENV PYTHONPATH=/install/lib/python3.12/site-packages \
    HF_HOME=/opt/models
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# ============================================================================
# Stage 2: Runtime — lean production image
# ============================================================================
FROM python:3.12-slim AS runtime

# APP_ENV=production makes the image fail-closed: app/config.py refuses to start
# without GOOGLE_API_KEY (which would otherwise silently serve mock answers).
# docker-compose overrides this to "development" for local runs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    HF_HOME=/opt/models \
    HF_HUB_OFFLINE=1

# curl — HEALTHCHECK probe; git — required by gitpython (loaders.py)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-built site-packages and the pre-downloaded model weights.
COPY --from=builder /install /usr/local
COPY --from=builder /opt/models /opt/models

WORKDIR /app

# Application source last: it changes most often, and Docker invalidates every
# layer below a changed one. Ordering least- to most-frequently-changed keeps
# the slow pip layers cached.
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY frontend/ ./frontend/

# Non-root user for runtime security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --no-create-home --shell /bin/false appuser && \
    mkdir -p /app/chroma_db /app/data && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /opt/models

USER appuser

EXPOSE 8000

# PORT is injected by Render/Railway; falls back to 8000 for local/Azure runs
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# --proxy-headers makes request.client.host reflect the real client behind a
# load balancer. Pair it with TRUST_PROXY_HEADERS=true so rate limiting buckets
# per user rather than per proxy.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers
