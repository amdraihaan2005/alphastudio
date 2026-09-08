# Alpha Studio

**Live Demo:** [https://alphastudio-murex.vercel.app/](https://alphastudio-murex.vercel.app/)

## Overview
Alpha Studio is a full-stack application structured into frontend, backend, and data processing modules.

## Architecture & Process Flow
The application operates in three main phases:

### Phase 1: Data Ingestion & Processing
- **Module:** `data/`
- Handles the automated downloading and parsing of required datasets.
- Prepares data for vectorization or relational storage.

### Phase 2: Core Backend Engine
- **Module:** `backend/`
- Built with FastAPI and structured using domain-driven design.
- Handles user authentication, database models (SQLAlchemy/Alembic), and exposes REST APIs.
- Features dedicated modules for AI assistant logic, chat websockets, RAG (Retrieval-Augmented Generation), and response grounding.

### Phase 3: Frontend Presentation
- **Module:** `frontend/`
- A modern UI built with Vite, React (implied by typical stacks), and Tailwind CSS.
- Communicates securely with the backend API to provide a dynamic and responsive user experience.

## Directory Structure

### Complete Repository
```text
Alpha Studio
├── backend/              # Core API and AI logic
├── frontend/             # UI source code and assets
├── data/                 # Data fetching and scripts
└── README.md
```

### Backend Breakdown (`backend/app/`)

The core engine is powered by FastAPI and organized into specific functional modules to ensure a clean, maintainable architecture:

- **`main.py` & `config.py`:** `main.py` acts as the entry point, initializing the FastAPI app, connecting routers, and starting services. `config.py` securely loads environment variables (like database URIs and API keys).
- **`auth/`:** Manages user authentication, token generation, and ensures that endpoints are protected from unauthorized access.
- **`chat/`:** The real-time conversational engine. It typically handles WebSockets for streaming chat messages between the user and the AI.
- **`assistant/`:** Houses the AI's core logic, prompt engineering, and behavioral constraints that define the assistant's persona.
- **`database/`:** Manages connections to the database and defines the ORM (Object-Relational Mapping) models using SQLAlchemy, handling how data like chat histories are saved and retrieved.
- **`ingest/`:** The data pipeline. It takes raw files (downloaded in Phase 1), chunks them into readable pieces, and vectorizes them for AI search.
- **`retrieval/`:** Implements RAG (Retrieval-Augmented Generation). When a user asks a question, this module queries the vectorized database to retrieve the most relevant context.
- **`grounding/`:** The fact-checking layer. It cross-references the AI's generated response against the retrieved context to minimize hallucinations and ensure accuracy.
- **`routers/` & `schemas/`:** The API Layer. `routers` define the RESTful URLs that the frontend interacts with. `schemas` (using Pydantic) validate all incoming and outgoing data for these routes.

```text
backend/app
├── main.py             # Application entry point & setup
├── config.py           # Environment variables & configuration
│
├── Modules:
│   ├── auth/           # Authentication & security
│   ├── chat/           # Core chat engine & WebSocket logic
│   ├── assistant/      # AI assistant logic & prompts
│   ├── database/       # DB connections & ORM models
│   ├── ingest/         # Data ingestion pipelines
│   ├── retrieval/      # RAG & vector search logic
│   └── grounding/      # Fact-checking & response grounding
│
└── API Layer:
    ├── routers/        # REST API endpoints & route definitions
    └── schemas/        # Pydantic data validation models
```
