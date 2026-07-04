# Use a lightweight official Python image
FROM python:3.12-slim-bookworm AS builder

# Install uv for rapid, caching builds
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency definition files
COPY pyproject.toml uv.lock /app/

# Install dependencies into project's virtual environment
RUN uv sync --frozen --no-install-project --no-dev

# Final runtime image
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY app /app/app
COPY alembic /app/alembic
COPY alembic.ini /app/

# Set PATH to use the virtual environment's binaries
ENV PATH="/app/.venv/bin:$PATH"

# Expose server port
EXPOSE 8000

# Start command (Railway dynamically sets $PORT)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
