FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# We install as the service user, since a chown -R afterwards copies the whole virtualenv into
# a new layer and doubles the image.
RUN adduser --system app && install -d -o app /app
USER app
ENV UV_NO_CACHE=1

WORKDIR /app

# Lockfile first, so dependency layers survive a source change.
COPY --chown=app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app . .
RUN uv sync --frozen --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD [".venv/bin/python", "-m", "midifier", "api"]
