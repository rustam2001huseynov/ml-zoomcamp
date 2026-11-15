# Build from the project folder (the dependency files are there):
#     docker build -f production/Dockerfile -t guitar-tab .
FROM python:3.12-slim

# uv installs the exact versions from uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /bin/uv

WORKDIR /app
# NUMBA_CACHE_DIR: librosa caches compiled code. If the container runs as a non-root user, it
# can't write to site-packages, so cache in /tmp instead.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    NUMBA_CACHE_DIR=/tmp/numba_cache

# Dependencies first, so this layer is cached when only the code changes.
# nvidia-nccl is only needed for multi-GPU XGBoost training; skipping it saves ~300 MB.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project --no-install-package nvidia-nccl-cu13

COPY production/guitar.py production/predict.py production/model.bin ./

EXPOSE 9696

# Port 9696 by default; hosts such as Render pass the port to use in $PORT.
CMD ["sh", "-c", "exec uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}"]
