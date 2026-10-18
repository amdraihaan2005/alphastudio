# Design Document: AsyncIO Event Loop Lifecycle & Shutdown Fixes

**Date**: 2026-10-18  
**Topic**: Resolve `RuntimeError: Event loop is closed` across API requests, thread pools, and Windows Uvicorn shutdown.

---

## 1. Problem Statement

During API requests and server operations, Python raises:
```text
File ".../asyncio/base_events.py", line 550, in _check_closed
    raise RuntimeError('Event loop is closed')
```

### Root Causes
1. **Secondary Thread Event Loop Poisoning in Chat Orchestration**:
   `orchestrate_chat_stream` wrapped synchronous `retrieve_hybrid` in `run_in_threadpool`. `retrieve_hybrid` in turn ran `asyncio.run(aretrieve_hybrid_raw(...))`, spinning up a temporary event loop in a worker thread and terminating it when done.
2. **Stale Singleton Async Clients**:
   `cohere.AsyncClientV2` and `ChatGroq` use `httpx.AsyncClient`, which binds its HTTP connection pool to the event loop active during first use. Because these clients were cached in persistent module-level globals (`_cohere_client`, `_groq_llm`), closing a loop left them holding references to closed loops.
3. **Unguarded Callbacks in CPU Chunk Ingestion**:
   In `_sync_produce_chunks`, `loop.call_soon_threadsafe()` was invoked without checking `loop.is_closed()`. If a consumer failed, timed out, or disconnected, the thread continued calling into a closed loop.
4. **Windows Proactor Teardown Noise**:
   On Windows with Python 3.8+, Uvicorn shutdown or hot-reload triggers `_ProactorBasePipeTransport.__del__` after the event loop is already marked closed.

---

## 2. Design Specification

### 2.1 Native Async Retrieval in Chat Orchestrator
- In `backend/app/chat/orchestrator.py`:
  Replace `await run_in_threadpool(retrieve_hybrid, ...)` with direct `await aretrieve_hybrid_raw(...)`.
  `aretrieve_hybrid_raw` natively uses `asyncio.to_thread` for CPU/DB queries while preserving the current ASGI event loop for async operations (`decompose_query`, `aembed_query`).

### 2.2 Native Async Agent Tool
- In `backend/app/assistant/agent.py`:
  Convert `@tool def search_filings(...)` to `@tool async def search_filings(...)`.
  Inside `search_filings`, directly await `aretrieve_hybrid_raw(...)` to eliminate thread-pool bridging when LangGraph executes agent turns.

### 2.3 Loop-Aware Client Lifecycle
- In `backend/app/ingest/embedder.py`:
  Track `_cohere_loop`. When `get_cohere_client()` is called, compare `asyncio.get_running_loop()` against `_cohere_loop`. Recreate `cohere.AsyncClientV2` if `_cohere_client` is None, `_cohere_loop is not current_loop`, or `_cohere_loop.is_closed()`.
- In `backend/app/retrieval/decomposer.py`:
  Track `_groq_llm_loop`. In `get_groq_llm()`, compare `asyncio.get_running_loop()` against `_groq_llm_loop`. Recreate `ChatGroq` if the loop changed or is closed.

### 2.4 Guarded Threadsafe Queue Dispatch
- In `backend/app/ingest/orchestrator.py`:
  Before invoking `loop.call_soon_threadsafe(queue.put_nowait, ...)`, check:
  `if not loop.is_closed() and not stop_event.is_set():`

### 2.5 Windows Proactor Teardown Suppression
- In `backend/app/main.py`:
  Intercept `_ProactorBasePipeTransport.__del__` on `win32` platform to suppress benign `RuntimeError: Event loop is closed` on shutdown/reload.

---

## 3. Data Flow Diagram

```
FastAPI ASGI Event Loop
       │
       ▼
orchestrate_chat_stream (Async Generator)
       │
       ├─► [await] aretrieve_hybrid_raw(db, user_query, ...)
       │        ├─► decompose_query (uses loop-aware ChatGroq)
       │        ├─► aembed_query (uses loop-aware Cohere AsyncClientV2)
       │        └─► asyncio.to_thread(_fetch_candidates, _fetch_neighbors)
       │
       ├─► [await] agent.run / ainvoke(user_query)
       │        └─► async search_filings tool (awaits aretrieve_hybrid_raw)
       │
       └─► SSE Tokens & Citations Streamed to Client
```

---

## 4. Verification Plan

1. **Test Suite Regression**: Run `uv run pytest` to ensure all 70 existing unit tests pass without failure.
2. **Loop Isolation Tests**: Run tests that invoke `get_cohere_client()` and `get_groq_llm()` across multiple `asyncio.run()` invocations to confirm clean client re-binding without runtime errors.
3. **Chat Stream Validation**: Validate that streaming endpoints execute without encountering event loop closed exceptions.
