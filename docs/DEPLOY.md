# MediBot — Deploy

> How to run locally and deploy. Three options: docker-compose (recommended for dev),
> Streamlit Cloud (fastest for demo), or a VPS/cloud VM.

---

## 1. Local dev (no Docker)

### Prerequisites
- Python 3.11+
- Qdrant running locally (`docker run -p 6333:6333 qdrant/qdrant`)
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
python scripts/ingest.py

# start backend
uvicorn src.api.app:app --reload --port 8000

# start frontend (new terminal)
streamlit run frontend/app.py --server.port 3000
```

---

## 2. Docker Compose (recommended)

```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - qdrant

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:8501"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      - backend

volumes:
  qdrant_data:
```

```bash
docker-compose up --build
# ingest (first time only):
docker-compose exec backend python scripts/ingest.py
```

### Dockerfiles

**Dockerfile.backend:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile.frontend:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY frontend/ frontend/
CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

---

## 3. Cloud deploy options

### Option A: Streamlit Community Cloud (free, fastest)

- Push repo to GitHub (public).
- Go to share.streamlit.io → deploy `frontend/app.py`.
- Set secrets (GROQ_API_KEY, BACKEND_URL) in Streamlit's secrets manager.
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
# .env.example
GROQ_API_KEY=your_groq_api_key
QDRANT_HOST=localhost          # or qdrant (in docker-compose)
QDRANT_PORT=6333
SQLITE_PATH=data/mediassist.db
JWT_SECRET=change-this-in-production
COLLECTION_NAME=medibot
LOG_LEVEL=INFO
```

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
      - run: python scripts/ingest.py
        env:
          QDRANT_HOST: localhost
      - run: pytest evals/ -v --tb=short
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          QDRANT_HOST: localhost
```
