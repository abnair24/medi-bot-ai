# MediBot — Deploy

> How to run locally and deploy. Three options: docker-compose (recommended for dev),
> Streamlit Cloud (fastest for demo), or a VPS/cloud VM.

---

## 1. Local dev (no Docker)

### Prerequisites
- Python 3.11+
- Qdrant running locally. Easiest is `docker compose up -d qdrant`, which maps host
  **6335 → 6333** (matches `QDRANT_PORT` in `.env`). If you run it standalone with
  `docker run -p 6335:6333 qdrant/qdrant`, keep `QDRANT_PORT=6335` in `.env`.
- Groq API key (free tier)

### Steps

```bash
# clone and setup
git clone https://github.com/YOUR_USER/medibot.git
cd medibot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# env
cp .env.example .env
# edit .env: GROQ_API_KEY=your_key

# ingest documents (run once)
python backend/ingestion/ingest.py

# start backend (host port 8000 for local dev). Backend code lives in backend/,
# so put it on PYTHONPATH for `from src.api.app` to resolve.
PYTHONPATH=backend uvicorn src.api.app:app --reload --port 8000

# start frontend (new terminal). Point it at the backend and use the project's
# configured port (8700 per .streamlit/config.toml).
MEDIBOT_API_URL=http://localhost:8000 streamlit run frontend/app.py --server.port 8700
```

---

## 2. Docker Compose (recommended)

The authoritative compose spec is the checked-in **`docker-compose.yml`** at the repo
root — do not copy it here (that copy always drifts). It defines four services:

| Service | Host port → container | Notes |
|---|---|---|
| `qdrant` | **6335** → 6333 | remapped off 6333 to avoid collisions; 6000 was browser-blocked (X11). Dashboard: `http://localhost:6335/dashboard` |
| `ingest` | (one-shot) | populates Qdrant from `backend/ingestion/data/docs/` **only if the collection is empty**, then exits. Runs `backend/ingestion/ensure_ingested.py` |
| `backend` | **8001** → 8000 | FastAPI; `EVALS_ENABLED=true`; healthcheck on `/health` |
| `frontend` | **8700** → 8700 | Streamlit; waits for `backend` to be `service_healthy` |

**Startup ordering** (all handled by compose conditions — no race):

```
qdrant → ingest (populate index, then exit)
       → backend (waits for ingest to COMPLETE, so retrieval works immediately)
       → frontend (waits for backend /health)
```

Evals are **not** run at startup — they run on demand from the admin **Evaluation** tab
(`POST /evals/run`). So `docker compose up` is fast and spends no LLM tokens; the source
docs and SQLite DB are committed, so a fresh clone ingests and works out of the box.

```bash
# build + start everything. First `up` runs ingestion (downloads embedding models,
# a few minutes), then backend + frontend come up with a populated index.
docker compose up -d --build

# force a re-ingest (e.g. after changing docs): wipe the volume, then up
# rm -rf qdrant_data && docker compose up -d
# ...or run ingestion manually against a running stack:
docker compose run --rm ingest python backend/ingestion/ingest.py

# logs
docker compose logs -f backend
docker compose logs ingest       # ingestion progress / "skip (already populated)"
```

To run the full eval suite headless (CI or a local sweep) instead of via the UI:

```bash
docker compose run --rm ingest python backend/ingestion/run_evals.py   # honours EVAL_TRACKS
```

Dockerfiles are `Dockerfile.backend` and `Dockerfile.frontend` at the repo root. The
backend image installs dev/eval deps (pytest + deepeval) via an **ordered pip install**
(dev first, then base) to resolve the deepeval↔huggingface-hub `click` version conflict —
see the header comment in `requirements-dev.txt`.

---

## 3. Cloud deploy options

### Option A: Streamlit Community Cloud (free, fastest)

- Push repo to GitHub (public).
- Go to share.streamlit.io → deploy `frontend/app.py`.
- Set secrets (GROQ_API_KEY, MEDIBOT_API_URL) in Streamlit's secrets manager.
- Backend + Qdrant need separate hosting (see option B).

### Option B: Railway / Render / Fly.io

- Deploy docker-compose (or individual services).
- Railway supports docker-compose natively.
- Qdrant Cloud has a free tier (1GB) — use instead of self-hosted for simplicity.
- Set env vars in the platform's dashboard.

### Option C: VPS (DigitalOcean / Hetzner)

- Cheapest for always-on: a $6/month VPS runs all three containers.
- `docker-compose up -d` and point a domain at it.

---

## 4. Environment variables

```bash
# .env.example  (see the committed .env.example for the full list)
GROQ_API_KEY=your_groq_api_key
QDRANT_HOST=localhost          # compose overrides this to the service name `qdrant`
QDRANT_PORT=6335               # host mapping; compose overrides to 6333 (container-internal)
SQLITE_PATH=backend/ingestion/data/mediassist.db
JWT_SECRET=change-this-in-production
COLLECTION_NAME=medibot
LOG_LEVEL=INFO
MEDIBOT_API_URL=http://localhost:8000   # frontend → backend; compose sets http://backend:8000
```

Note: in `docker-compose.yml` the backend service overrides `QDRANT_HOST=qdrant` and
`QDRANT_PORT=6333` (inter-container traffic uses Qdrant's internal port, not the host
remap), and the frontend sets `MEDIBOT_API_URL=http://backend:8000`.

**Important:** generate a real JWT secret before deploying outside local dev:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

The backend includes a startup guard: if `JWT_SECRET` is the default value and the
`ENVIRONMENT` variable is not `dev`, the app refuses to start. This prevents accidental
deployment with a known secret.

---

## 5. Running evals in CI

The eval suite needs a running Qdrant with ingested data. In GitHub Actions:

```yaml
# .github/workflows/eval.yml
name: Eval Gate
on: [push, pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    services:
      qdrant:
        image: qdrant/qdrant
        ports:
          - 6333:6333
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python backend/ingestion/ingest.py
        env:
          QDRANT_HOST: localhost
      - run: pytest backend/evals/ -v --tb=short
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          QDRANT_HOST: localhost
```
