FROM python:3.12-slim-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into project's virtual environment
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source code
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

# Expose server port
EXPOSE 8000

# Start command using uv run (perfectly handles pathing and virtual envs)
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
