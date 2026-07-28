# --- build stage: resolve and install dependencies with uv ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY . .

# --- runtime stage: slim image, non-root user ---
FROM python:3.12-slim-bookworm
WORKDIR /app
RUN groupadd -r app && useradd -r -g app -m -d /home/app app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 HOME=/home/app
USER app
EXPOSE 8501
CMD ["sh", "-c", "streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]

