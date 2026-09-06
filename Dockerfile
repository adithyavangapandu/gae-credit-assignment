FROM ghcr.io/astral-sh/uv:0.12.10 AS uv
FROM python:3.11.16-slim-bookworm AS runtime

ARG INSTALL_GCP=false
ARG GIT_SHA=unknown
ENV GAE_GIT_SHA=$GIT_SHA

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN if [ "$INSTALL_GCP" = "true" ]; then uv sync --locked --extra gcp --no-dev --no-editable; \
    else uv sync --locked --no-dev --no-editable; fi
COPY configs ./configs
COPY scripts ./scripts
RUN groupadd --gid 10001 experiment \
    && useradd --uid 10001 --gid experiment --create-home experiment \
    && mkdir -p /app/runs \
    && chown -R experiment:experiment /app/runs
USER experiment
ENTRYPOINT ["python", "-m", "gae_credit.train"]
CMD ["--config", "configs/smoke_dense_h3.yaml"]

FROM runtime AS test
USER root
RUN uv sync --locked --extra gcp --no-editable
COPY tests ./tests
RUN uv run --extra gcp ruff check . && uv run --extra gcp pytest

FROM runtime AS final
