# Install Poetry and create venv in the builder step,
# then copy the venv to the runtime image, so that the runtime image is as small as possible.
FROM --platform=linux/amd64 python:3.10.14-bookworm AS builder

RUN pip install poetry==1.8.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root

FROM --platform=linux/amd64 python:3.10.14-bookworm AS runtime

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY pyproject.toml poetry.lock ./
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

CMD ["bash", "-c", "python prediction_market_agent/run_agent.py ${runnable_agent_name} ${market_type}"]
