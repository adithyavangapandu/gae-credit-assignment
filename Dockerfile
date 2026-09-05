FROM ghcr.io/astral-sh/uv:0.12.10 AS uv
FROM python:3.11.16-slim-bookworm

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable
COPY configs ./configs
COPY scripts ./scripts
RUN groupadd --gid 10001 experiment \
    && useradd --uid 10001 --gid experiment --create-home experiment \
    && mkdir -p /app/runs \
    && chown -R experiment:experiment /app/runs
USER experiment
ENTRYPOINT ["python", "-m", "gae_credit.train"]
CMD ["--config", "configs/smoke_dense_h3.yaml"]
