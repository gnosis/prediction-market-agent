# Install Poetry and create venv in the builder step,
# then copy the venv to the runtime image, so that the runtime image is as small as possible.
FROM --platform=linux/amd64 python:3.10.14-slim-bookworm AS builder

RUN pip install poetry==1.8.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root --only main

FROM --platform=linux/amd64 python:3.10.14-slim-bookworm AS runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY prediction_market_agent ./prediction_market_agent

ENV PYTHONPATH=/app

# TODO: This is a hotfix, because we are unable to lock this version with mech-client, remove this ASAP when PRs are merged into Valory and update pyproject in PMAT.
# This also works locally, after doing `poetry install` just go to `poetry shell` and run `pip install crewai["tools"]==0.22.5`.
RUN pip install 'crewai[tools]'==0.22.5

CMD ["bash", "-c", "python prediction_market_agent/run_agent.py ${runnable_agent_name} ${market_type}"]
