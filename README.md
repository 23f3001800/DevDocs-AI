# DevDocs AI

**RAG over any GitHub repo or docs site — built for developers who want instant, grounded answers from their codebase.**

[![CI/CD](https://github.com/23f3001800/DevDocs-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/23f3001800/DevDocs-AI/actions)
[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](Dockerfile)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The Problem

Developers waste **hours** searching through docs, README files, and source code to find answers. Existing search tools return keyword matches — not answers. LLMs hallucinate when they don't have your specific docs.

**DevDocs AI** solves this by combining:
- **Hybrid retrieval** (dense + sparse + reranking) for precision
- **Gemini** for grounded, citation-backed answers
- **SSE streaming** for instant time-to-first-token
- **RAGAS evaluation** to prove it actually works

---

## Key Features

| Feature | Implementation |
|---------|---------------|
| 🔍 **Hybrid Search** | Dense embeddings (MiniLM) + BM25 sparse + RRF fusion |
| 🎯 **Cross-Encoder Reranking** | ms-marco-MiniLM-L-6-v2 for final precision |
| ⚡ **Streaming Answers** | Server-Sent Events, real-time token delivery (~300ms TTFT) |
| 🔒 **RBAC Auth** | JWT + bcrypt with user/admin roles |
| 📊 **RAGAS Eval Gate** | Faithfulness, Relevancy, Precision — CI blocks deploys below 0.75 |
| 🐳 **Production Docker** | Multi-stage, non-root, HEALTHCHECK, CPU-only torch, models baked in |
| 📈 **Observability** | LangSmith tracing + embedding cache metrics |
| 🚀 **Zero Cold Start** | Models baked into the image + pre-warmed on startup |

---

## Architecture

```mermaid
graph TB
    subgraph Ingestion
        A[GitHub Repo / URL / PDF] --> B[Loaders]
        B --> C[Language-Aware Chunker]
        C --> D[SentenceTransformer Embed]
        D --> E[(ChromaDB)]
    end

    subgraph Query Pipeline
        F[User Question] --> G[FastAPI + RBAC Auth]
        G --> H{Hybrid Retrieval}
        H --> I[Dense Search - MiniLM]
        H --> J[BM25 Sparse Search]
        I --> K[RRF Merge]
        J --> K
        K --> L[CrossEncoder Rerank]
        L --> M[Gemini - Streaming]
        M --> N[Grounded Answer + Sources]
    end

    subgraph Evaluation
        O[Golden Dataset - 15Q] --> P[RAGAS]
        P --> Q{Avg ≥ 0.75?}
        Q -->|Yes| R[Deploy ✅]
        Q -->|No| S[Block ❌]
    end

    E -.-> I
    E -.-> J
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/23f3001800/DevDocs-AI.git
cd DevDocs-AI
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your keys:
#   GOOGLE_API_KEY=...
#   JWT_SECRET=your-random-secret  (openssl rand -hex 32)
```

### 3. Ingest Documentation

```bash
# Ingest a GitHub repo
python scripts/ingest.py --source https://github.com/tiangolo/fastapi

# Ingest a docs URL
python scripts/ingest.py --source https://docs.python.org/3/tutorial/
```

### 4. Run

```bash
uvicorn app.main:app --reload
# Open http://localhost:8000 for the chat UI
# API docs at http://localhost:8000/docs
```

### Docker (Recommended)

```bash
docker compose up --build
# Or standalone:
docker build -t devdocs-ai . && docker run -p 8000:8000 --env-file .env devdocs-ai
```

### Production configuration

The Docker image sets `APP_ENV=production`, which makes the app **fail-closed** —
it refuses to start unless these are set explicitly:

| Variable | Why it's required |
|----------|-------------------|
| `JWT_SECRET` | Without it, tokens are signed with a public constant and anyone can forge an admin JWT. Generate with `openssl rand -hex 32`. |
| `ADMIN_PASSWORD` | Seeds the auto-created `admin` account. Refuses to fall back to a known default. |
| `GOOGLE_API_KEY` | Without it the app would silently answer with the mock provider while `/health` stayed green. |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins. Defaults to `*` — pin to your frontend origin. |

`docker compose` overrides `APP_ENV` to `development` for local runs, so the
above are optional when developing.

---

## API Reference

### Auth

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/auth/register` | POST | `{username, password}` | Create account → JWT |
| `/auth/login` | POST | `{username, password}` | Login → JWT |
| `/auth/me` | GET | — | Current user info |
| `/auth/logout` | POST | — | Revoke the presented token (server-side denylist) |

### Query (requires `user` role)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create a POST endpoint in FastAPI?", "k": 5}'
```

Responses stream as **Server-Sent Events** with three event types:
`token` (`{"text": "..."}`), `sources` (`{"sources": [...]}`), then `done`.
A failure mid-stream arrives as an `error` event — once streaming starts the
HTTP 200 is already committed, so errors can only be delivered in-band.

### Ingest (requires `user` role)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Queue an ingest → **202** `{job_id}` |
| `/ingest/{job_id}` | GET | Poll job status (`queued`/`running`/`succeeded`/`failed`) |

Ingestion is asynchronous because cloning + chunking + embedding a real
repository takes minutes, and load balancers cut idle connections at ~230s.
Any authenticated user may ingest; every submitted URL passes an **SSRF guard**
(private / loopback / link-local / cloud-metadata addresses are rejected) before
the server makes any network call. Deleting a source stays admin-only.

### Admin (requires `admin` role)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sources` | DELETE | Purge every chunk from a source |
| `/metrics` | GET | Operational metrics + cache stats |
| `/admin/users` | GET | List all registered users |

### Ops (public)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + chunk count |

---

## RAGAS Evaluation Scores

Every push to `main` triggers the RAGAS eval gate in CI. Deploys are blocked if the average score falls below **0.75**.

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Faithfulness | ≥ 0.75 | Is the answer grounded in the retrieved context? |
| Answer Relevancy | ≥ 0.75 | Does the answer address the question? |
| Context Precision | ≥ 0.75 | Are the retrieved chunks relevant? |

```bash
# Generate scores locally
python -m scripts.run_evals
# Results → docs/evaluation/ragas_results.json
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM** | Gemini (`gemini-2.5-flash`) | Fast, cheap, long context; single-provider by design |
| **Embeddings** | all-MiniLM-L6-v2 | Fast, 384-dim, great quality/speed tradeoff |
| **Vector DB** | ChromaDB | Persistent, embedded, zero-config |
| **Sparse Search** | BM25 (rank_bm25) | Catches keyword matches that dense embeddings miss |
| **Reranker** | CrossEncoder ms-marco | Final precision gate — 10x more accurate than bi-encoder |
| **Framework** | FastAPI | Async, streaming, auto OpenAPI docs |
| **Auth** | JWT + bcrypt | Stateless, no session store needed |
| **Eval** | RAGAS + `gemini-2.5-pro` judge | Stronger model judges the answering model |
| **Tracing** | LangSmith | Full observability for LLM calls |
| **CI/CD** | GitHub Actions | lint → test → eval gate → Docker → Azure |
| **Deploy** | Docker + Azure App Service | Multi-stage, non-root, CPU-only torch (~1.5 GB saved) |

---

## Project Structure

```
DevDocs-AI/
├── app/
│   ├── main.py              # FastAPI app — routes, RBAC, middleware
│   ├── chain.py             # RAG chain — retrieval → Gemini → response
│   ├── hybrid_retriever.py  # Dense + BM25 + RRF + CrossEncoder
│   ├── vectorstore.py       # ChromaDB + embedding cache
│   ├── bm25_retriever.py    # BM25 sparse search
│   ├── chunker.py           # Language-aware text splitting
│   ├── loaders.py           # GitHub, URL, PDF document loaders
│   ├── models.py            # Pydantic response schemas
│   ├── auth.py              # JWT auth + RBAC (user/admin)
│   ├── config.py            # Validated settings + fail-closed prod checks
│   ├── database.py          # SQLite users + JWT revocation denylist
│   └── retriever_instance.py # Shared singleton + pre-warm
├── frontend/
│   ├── index.html           # Chat UI
│   ├── styles.css           # Dark mode + glassmorphism
│   └── app.js               # SSE client + markdown render
├── scripts/
│   ├── ingest.py            # CLI ingestion tool
│   └── run_evals.py         # RAGAS evaluation harness
├── tests/
│   ├── conftest.py          # Temp DB/Chroma isolation for the whole suite
│   ├── test_api.py          # API + RBAC integration tests
│   ├── test_units.py        # Offline unit tests
│   ├── test_retrieval.py    # RRF fusion + rerank ordering + chunk IDs
│   ├── test_stream.py       # SSE framing + stream fallback
│   └── golden_set.json      # 15-question eval dataset
├── .github/workflows/ci.yml # CI/CD pipeline
├── Dockerfile               # Multi-stage production build
├── docker-compose.yml       # One-command deployment
├── requirements.txt         # Python dependencies
└── pyproject.toml           # Ruff + pytest config
```

---

## Known Limitations

Stated plainly rather than discovered in production:

- **`/metrics` and in-memory rate limiting are per-process.** With more than one
  uvicorn worker or replica, `/metrics` reports whichever process served the
  request, and a `30/minute` limit becomes `30 × workers`. Set
  `RATE_LIMIT_STORAGE_URI=redis://...` to share the limiter; scrape Prometheus
  if you need cross-replica metrics.
- **Ingest jobs live in process memory.** A status poll that lands on a
  different worker returns 404. A shared store or a real task queue is the fix
  once you scale out.
- **SQLite is on the container filesystem.** On a PaaS without a mounted volume
  every redeploy wipes the user table and re-seeds `admin`. Mount persistent
  storage at `/app/data`, or move to Postgres, before treating accounts as
  durable.
- **The JWT lives in `localStorage`.** Any XSS on the page can read it. Logout
  is real (server-side `jti` denylist), but moving to an `httpOnly` + `SameSite`
  cookie — with CSRF protection — is the stronger posture.
- **The SSRF guard has a DNS-rebinding window.** The address is validated, then
  resolved again by `requests`/`git` when the fetch happens. Redirects are
  refused for the same reason. `/ingest` is open to any authenticated `user`,
  so this guard is now the primary control — pinning the resolved IP through
  the fetch (closing the rebinding window) is the stronger posture.
- **RAGAS judges Gemini with Gemini.** The judge defaults to a stronger model
  (`gemini-2.5-pro`) than the one answering (`gemini-2.5-flash`), which reduces
  self-preference bias but does not eliminate it. Read the scores as a
  regression signal over time, not an absolute quality measure.

---

## Roadmap

- [ ] 💬 **Conversation memory** — multi-turn chat with context window
- [ ] 📚 **Multi-repo support** — switch between ingested repos
- [ ] 🔑 **OAuth providers** — GitHub/Google SSO
- [ ] 📊 **Analytics dashboard** — query patterns, popular docs
- [ ] 🧪 **A/B testing** — compare retrieval strategies with RAGAS

---

## License

MIT — see [LICENSE](LICENSE) for details.
