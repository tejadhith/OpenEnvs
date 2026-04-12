# Multi-stage build for the shader environment.
# Requires EGL for headless OpenGL rendering.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=shader

COPY . /app/env

WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# EGL support for headless OpenGL rendering
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libegl1 libgl1 libgles2 libgl1-mesa-dri && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Download shaders21k corpus (skip if data/ already present from COPY)
RUN if [ ! -d /app/env/data/shadertoy ]; then \
        pip install --no-cache-dir gdown && \
        gdown "1kIiBdeW9CEIfRlYOYTuxfWvN036k3Iig" -O /tmp/shaders.zip && \
        python3 -c "import zipfile; zipfile.ZipFile('/tmp/shaders.zip').extractall('/app/env')" && \
        mv /app/env/shader_codes /app/env/data && \
        rm /tmp/shaders.zip; \
    fi

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
