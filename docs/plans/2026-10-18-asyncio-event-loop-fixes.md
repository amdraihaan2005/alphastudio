# AsyncIO Event Loop Lifecycle & Shutdown Fixes Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Eliminate `RuntimeError: Event loop is closed` during active chat stream/upload requests and Windows server shutdown by replacing thread-to-async bridges with native async calls, making client getters loop-aware, and guarding background callback dispatch.

**Architecture:** Chat retrieval and the agent's `search_filings` tool will directly await `aretrieve_hybrid_raw` on the running ASGI loop. `get_cohere_client()` and `get_groq_llm()` will inspect `asyncio.get_running_loop()` and dynamically re-instantiate clients if the loop changed or closed. The background chunk producer will verify `not loop.is_closed()` before threadsafe callbacks.

**Tech Stack:** Python 3.13, FastAPI, Starlette, AsyncIO, LangChain, LangGraph, ChatGroq, Cohere AsyncClientV2, Pytest.

---

### Task 1: Loop-Aware Cohere Async Client Getter

**Files:**
- Modify: `backend/app/ingest/embedder.py:35-53`
- Test: `backend/tests/test_embedder.py`

**Step 1: Write the failing test**

Add to `backend/tests/test_embedder.py`:
```python
def test_cohere_client_recreated_when_loop_changes():
    import asyncio
    from app.ingest.embedder import get_cohere_client
    import app.ingest.embedder as embedder_module

    embedder_module._cohere_client = None
    embedder_module._cohere_loop = None

    async def run_in_loop():
        return get_cohere_client()

    client1 = asyncio.run(run_in_loop())
    client2 = asyncio.run(run_in_loop())

    assert client1 is not client2, "Client should be re-instantiated when event loop closes/changes"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_embedder.py -k test_cohere_client_recreated_when_loop_changes -v`  
Expected: FAIL with `AssertionError: Client should be re-instantiated when event loop closes/changes`

**Step 3: Write minimal implementation**

In `backend/app/ingest/embedder.py`:
```python
# Lazy-loaded singleton holders
_cohere_client: cohere.AsyncClientV2 | None = None
_cohere_loop: asyncio.AbstractEventLoop | None = None


def get_cohere_client() -> cohere.AsyncClientV2:
    """
    Loop-aware lazy-loading client getter function.
    Recreates AsyncClientV2 if the active event loop changes or has been closed,
    preventing 'RuntimeError: Event loop is closed' across threads and ASGI requests.
    """
    global _cohere_client, _cohere_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if (
        _cohere_client is None
        or _cohere_loop is not current_loop
        or (_cohere_loop and _cohere_loop.is_closed())
    ):
        if not settings.COHERE_API_KEY or not settings.COHERE_API_KEY.strip():
            raise ValueError(
                "COHERE_API_KEY is not configured in settings/environment."
            )
        _cohere_client = cohere.AsyncClientV2(api_key=settings.COHERE_API_KEY)
        _cohere_loop = current_loop
    return _cohere_client
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_embedder.py -v`  
Expected: PASS

**Step 5: Commit**

```powershell
git add backend/app/ingest/embedder.py backend/tests/test_embedder.py; git commit -m "fix(embedder): make get_cohere_client loop-aware to prevent closed loop errors"
```

---

### Task 2: Loop-Aware Groq LLM Getter

**Files:**
- Modify: `backend/app/retrieval/decomposer.py:40-56`
- Test: `backend/tests/test_retriever.py`

**Step 1: Write the failing test**

Add to `backend/tests/test_retriever.py`:
```python
def test_groq_llm_recreated_when_loop_changes():
    import asyncio
    from app.retrieval.decomposer import get_groq_llm
    import app.retrieval.decomposer as decomposer_module

    decomposer_module._groq_llm = None
    decomposer_module._groq_llm_loop = None

    async def run_in_loop():
        return get_groq_llm()

    llm1 = asyncio.run(run_in_loop())
    llm2 = asyncio.run(run_in_loop())

    assert llm1 is not llm2, "ChatGroq LLM should be re-instantiated when event loop closes/changes"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_retriever.py -k test_groq_llm_recreated_when_loop_changes -v`  
Expected: FAIL with `AssertionError: ChatGroq LLM should be re-instantiated when event loop closes/changes`

**Step 3: Write minimal implementation**

In `backend/app/retrieval/decomposer.py`:
```python
_groq_llm: ChatGroq | None = None
_groq_llm_loop: Any = None


def get_groq_llm(temperature: float = 0.0) -> ChatGroq:
    """Loop-aware singleton getter for the official GroqCloud ChatGroq client."""
    global _groq_llm, _groq_llm_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if (
        _groq_llm is None
        or _groq_llm_loop is not current_loop
        or (_groq_llm_loop and _groq_llm_loop.is_closed())
    ):
        if not settings.GROQ_API_KEY or not settings.GROQ_API_KEY.strip():
            raise ValueError("GROQ_API_KEY is not configured in settings/environment.")
        _groq_llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_LLM_MODEL,
            temperature=temperature,
        )
        _groq_llm_loop = current_loop
    return _groq_llm
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_retriever.py -v`  
Expected: PASS

**Step 5: Commit**

```powershell
git add backend/app/retrieval/decomposer.py backend/tests/test_retriever.py; git commit -m "fix(retrieval): make get_groq_llm loop-aware"
```

---

### Task 3: Native Async Retrieval in Chat Orchestrator

**Files:**
- Modify: `backend/app/chat/orchestrator.py:23,74-83`
- Test: `backend/tests/test_chat_orchestrator.py`

**Step 1: Write the failing test**

In `backend/tests/test_chat_orchestrator.py`, assert that `aretrieve_hybrid_raw` is awaited directly rather than calling `run_in_threadpool(retrieve_hybrid, ...)`:
```python
@pytest.mark.anyio
async def test_orchestrator_calls_native_aretrieve_hybrid():
    from app.chat.orchestrator import orchestrate_chat_stream
    import uuid
    from unittest.mock import patch, AsyncMock

    mock_aretrieve = AsyncMock(return_value=[])

    with patch("app.chat.orchestrator.aretrieve_hybrid_raw", mock_aretrieve):
        with patch("app.chat.orchestrator.run_in_threadpool", side_effect=AssertionError("Should not call run_in_threadpool for retrieval")):
            # Just generate first item from generator
            stream = orchestrate_chat_stream(uuid.uuid4(), "test query", user_id=uuid.uuid4())
            try:
                await anext(stream)
            except (StopAsyncIteration, AssertionError) as e:
                if isinstance(e, AssertionError) and "Should not call run_in_threadpool for retrieval" in str(e):
                    raise
    assert mock_aretrieve.called
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_chat_orchestrator.py -k test_orchestrator_calls_native_aretrieve_hybrid -v`  
Expected: FAIL with `AssertionError: Should not call run_in_threadpool for retrieval` or `AttributeError`

**Step 3: Write minimal implementation**

In `backend/app/chat/orchestrator.py`:
Replace:
```python
from app.retrieval.retriever import retrieve_hybrid
```
with:
```python
from app.retrieval.retriever import aretrieve_hybrid_raw
```
And replace:
```python
        # Non-blocking hybrid retrieval
        retrieved_chunks = await run_in_threadpool(
            retrieve_hybrid,
            db,
            user_query,
            user_id=user_id,
            limit=5,
            fetch_neighbors=True,
        )
```
with:
```python
        # Non-blocking hybrid retrieval (natively async without thread-loop bridging)
        retrieved_chunks = await aretrieve_hybrid_raw(
            db=db,
            query_text=user_query,
            user_id=user_id,
            limit=5,
            fetch_neighbors=True,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_chat_orchestrator.py -v`  
Expected: PASS

**Step 5: Commit**

```powershell
git add backend/app/chat/orchestrator.py backend/tests/test_chat_orchestrator.py; git commit -m "fix(chat): use native async aretrieve_hybrid_raw in orchestrator"
```

---

### Task 4: Native Async LangChain Agent Tool `search_filings`

**Files:**
- Modify: `backend/app/assistant/agent.py:34,170-215`
- Test: `backend/tests/test_agent.py`

**Step 1: Write the failing test**

In `backend/tests/test_agent.py`, add a test to verify `search_filings` can be invoked asynchronously via `ainvoke`:
```python
@pytest.mark.anyio
async def test_search_filings_tool_is_native_async():
    from app.assistant.agent import search_filings
    import inspect

    assert inspect.iscoroutinefunction(search_filings.coroutine) or inspect.iscoroutinefunction(search_filings.func), "search_filings tool must be a coroutine function"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_agent.py -k test_search_filings_tool_is_native_async -v`  
Expected: FAIL with `AssertionError: search_filings tool must be a coroutine function`

**Step 3: Write minimal implementation**

In `backend/app/assistant/agent.py`:
Import `aretrieve_hybrid_raw`:
```python
from app.retrieval.retriever import aretrieve_hybrid_raw, retrieve_hybrid
```
Update `search_filings` definition:
```python
@tool
async def search_filings(
    query: str,
    runtime: ToolRuntime[DocumentAgentDeps],
    filenames: list[str] | None = None,
) -> str:
    """
    Performs a supplementary hybrid search (semantic + full-text) on the database for a new query
    and returns matching passages with their document source name and page numbers.
    Optionally filter by specific document filenames for multi-file comparisons.
    """
    logger.info(f"Agent invoked search_filings tool for query: '{query}', filenames: {filenames}")
    deps = runtime.context if runtime and hasattr(runtime, "context") else None
    if not deps or not deps.db:
        return "Search error: No database session available in execution context."

    try:
        results = await aretrieve_hybrid_raw(
            db=deps.db,
            query_text=query,
            user_id=deps.user_id,
            filenames=filenames,
            limit=4,
            fetch_neighbors=False,
        )
        if not results:
            return "No matching filings found for this query."

        deps.retrieved_chunks.extend(results)

        formatted = []
        for idx, r in enumerate(results):
            c = r["chunk"]
            fname = c.document.filename if c.document else "unknown.pdf"
            formatted.append(
                f"Result {idx + 1}:\n"
                f"Database Chunk ID (only for tool usage): {c.id}\n"
                f"Source Filename (FILENAME for citation): {fname}\n"
                f"Page: {c.page_number}\n"
                f"Content: {c.text_content}\n"
                f"---"
            )
        return "\n\n".join(formatted)
    except Exception as err:
        logger.error(f"Agent tool search_filings failed: {err}", exc_info=True)
        return f"Search error occurred: {err}"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_agent.py -v`  
Expected: PASS

**Step 5: Commit**

```powershell
git add backend/app/assistant/agent.py backend/tests/test_agent.py; git commit -m "fix(agent): convert search_filings tool to native async"
```

---

### Task 5: Ingest Producer Loop-Closed Check

**Files:**
- Modify: `backend/app/ingest/orchestrator.py:73-90`
- Test: `backend/tests/test_orchestrator.py`

**Step 1: Write the failing test**

In `backend/tests/test_orchestrator.py`, verify `_sync_produce_chunks` handles a closed loop without raising `RuntimeError: Event loop is closed`:
```python
def test_sync_produce_chunks_with_closed_loop():
    import asyncio
    import threading
    from app.ingest.orchestrator import _sync_produce_chunks

    loop = asyncio.new_event_loop()
    queue = asyncio.Queue()
    stop_event = threading.Event()
    loop.close()  # Simulate loop closed while worker is running

    # Should not raise RuntimeError('Event loop is closed')
    _sync_produce_chunks(
        loop=loop,
        queue=queue,
        stop_event=stop_event,
        file_bytes=b"%PDF-1.4 dummy",
        filename="test.pdf",
    )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_orchestrator.py -k test_sync_produce_chunks_with_closed_loop -v`  
Expected: FAIL with `RuntimeError: Event loop is closed`

**Step 3: Write minimal implementation**

In `backend/app/ingest/orchestrator.py`, wrap callbacks in a loop-closed check:
```python
        if batch and not stop_event.is_set() and not loop.is_closed():
            loop.call_soon_threadsafe(queue.put_nowait, (batch, None))

        if not stop_event.is_set() and not loop.is_closed():
            # Push EOF sentinel: (None, None)
            loop.call_soon_threadsafe(queue.put_nowait, (None, None))

    except Exception as exc:
        if not stop_event.is_set() and not loop.is_closed():
            logger.warning(
                f"Error in sync chunk producer for '{filename}': {exc}"
            )
            loop.call_soon_threadsafe(queue.put_nowait, (None, exc))
```
And inside the chunk loop:
```python
            if len(batch) >= batch_size:
                if not loop.is_closed() and not stop_event.is_set():
                    loop.call_soon_threadsafe(queue.put_nowait, (batch, None))
                batch = []
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_orchestrator.py -v`  
Expected: PASS

**Step 5: Commit**

```powershell
git add backend/app/ingest/orchestrator.py backend/tests/test_orchestrator.py; git commit -m "fix(ingest): prevent calling closed loop in _sync_produce_chunks"
```

---

### Task 6: Windows Proactor Teardown Suppression in `main.py`

**Files:**
- Modify: `backend/app/main.py:1-35`
- Test: `backend/tests/test_agent.py`

**Step 1: Write implementation**

In `backend/app/main.py`:
```python
import sys

# Silence harmless CPython Windows Proactor shutdown cleanup errors on exit/reload
if sys.platform == "win32":
    from functools import wraps
    from asyncio.proactor_events import _ProactorBasePipeTransport

    _orig_del = _ProactorBasePipeTransport.__del__

    @wraps(_orig_del)
    def _silent_del(self, *args, **kwargs):
        try:
            _orig_del(self, *args, **kwargs)
        except RuntimeError as e:
            if str(e) != "Event loop is closed":
                raise

    _ProactorBasePipeTransport.__del__ = _silent_del
```

**Step 2: Run full regression test suite**

Run: `uv run pytest backend/tests`  
Expected: All tests PASS with code 0.

**Step 3: Commit**

```powershell
git add backend/app/main.py; git commit -m "fix(main): suppress Windows Proactor shutdown noise"
```
