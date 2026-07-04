# Vault CoPilot — Financial Document Analysis Platform

Vault CoPilot is a secure RAG (Retrieval-Augmented Generation) copilot that parses, chunks, embeds, and indexes Indian NSE annual report filings (Reliance Industries Limited and Tata Consultancy Services Limited) using dense-sparse hybrid search (pgvector + full-text indexing), reciprocal rank fusion (RRF), and PydanticAI.

---

## 🏗️ Tech Stack

- **Backend**: Python 3.12+, FastAPI, SQLAlchemy, Alembic, PostgreSQL (Supabase), Cohere v3 (embeddings), Groq (LLM inference), PydanticAI (grounded reasoning agent), Structlog (observability).
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui, Vercel AI SDK.

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
Ensure you have the following toolchains installed:
- [Python 3.12 or 3.13](https://www.python.org/downloads/)
- [`uv` package manager](https://github.com/astral-sh/uv) (strongly recommended)
- [Node.js 20+](https://nodejs.org/)
- [`pnpm` package manager](https://pnpm.io/)

### 2. Environment Configuration

#### Backend Setup (`backend/`)
Create a `backend/.env` file with the following variables:
```env
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-public-anon-key"
SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"

DATABASE_URL = "postgresql://postgres.your-project:password@host:5432/postgres"

GROQ_API_KEY = "gsk_..."
GROQ_LLM_MODEL = "llama3-70b-8192" # or model of your choice
COHERE_API_KEY = "cohere_..."
EMBEDDING_MODEL = "embed-english-v3.0"
EMBEDDING_DIMENSIONS = 1024
```

#### Frontend Setup (`frontend/`)
Create a `frontend/.env` file with the following variables:
```env
VITE_SUPABASE_URL = "https://your-project.supabase.co"
VITE_SUPABASE_ANON_KEY = "your-public-anon-key"
VITE_API_BASE_URL = "http://localhost:8000"
```

### 3. Ingesting Report Files (Corpus Seed)
1. Download 1-3 annual report PDFs (like `RELIANCE_2025.pdf`, `TCS_2025.pdf`) and place them in the `data/downloads/` directory.
2. In the root directory, configure the documents manifest:
   ```bash
   python data/download.py
   ```
3. Run the ingestion pipeline to parse, chunk, embed, and store context segments in your Supabase DB:
   ```bash
   cd backend
   uv run python -m app.ingest.pipeline
   ```

### 4. Running the Application

#### Start the Backend Server:
```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```
API docs will be available at `http://localhost:8000/docs`.

#### Start the Frontend Server:
```bash
cd frontend
pnpm install
pnpm dev
```
Open `http://localhost:5173` in your web browser.

---

## 🧪 Testing & Verification

### Running Automated Tests
To run backend unit and integration tests:
```bash
cd backend
uv run pytest
```

### Running the Smoke Test Suite
To run the programmatic smoke test checking RAG retrieval latency and grounding facts:
```bash
cd backend
uv run python smoke_test.py
```

---

## 🌐 Production Deployment

### 1. Database Setup
Ensure that vector extensions and RLS policies are enabled on the target database, then upgrade schemas:
```bash
cd backend
uv run alembic upgrade head
```

### 2. FastAPI Backend Deployment (e.g. Railway)
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Configure all backend `.env` keys in the host environment configuration dashboard.

### 3. Frontend SPA Deployment (e.g. Vercel / Netlify)
- Build Command: `pnpm build`
- Output Directory: `dist`
- Configure `VITE_*` keys as build-time variables.
