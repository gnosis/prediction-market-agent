# Install uv and create venv in the builder step,
# then copy the venv to the runtime image, so that the runtime image is as small as possible.
FROM --platform=linux/amd64 python:3.11.9-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Set to "true" to include dev dependencies (e.g., for running tests in Docker)
ARG INSTALL_DEV=false

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_DEV" = "true" ]; then \
    uv sync --frozen --no-install-project; \
    else \
    uv sync --frozen --no-install-project --no-dev; \
    fi

FROM --platform=linux/amd64 python:3.11.9-bookworm AS runtime

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY pyproject.toml uv.lock ./
COPY prediction_market_agent prediction_market_agent
COPY scripts scripts
COPY tests tests
COPY tokenizers tokenizers

ENV PYTHONPATH=/app
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Disable warning on importing `transformers`, as we don't use torch in this
# project and only intend to use it for tokenization. See:
# https://github.com/huggingface/transformers/issues/27214#issuecomment-1983731040
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV LANGFUSE_DEPLOYMENT_VERSION=none

CMD ["bash", "-c", "${VIRTUAL_ENV}/bin/python prediction_market_agent/run_agent.py ${runnable_agent_name} ${market_type}"]
