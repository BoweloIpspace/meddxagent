FROM python:3.12-slim AS builder

WORKDIR /src
RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY ddxdriver ./ddxdriver
RUN uv build --wheel


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app
RUN python -m pip install --no-cache-dir uv

COPY --from=builder /src/dist/*.whl /tmp/
RUN uv pip install --system torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --system /tmp/*.whl \
    && rm -f /tmp/*.whl \
    && useradd --create-home --uid 10001 meddx

USER meddx

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/api/v1/health', timeout=3)"]

CMD ["meddx-clinical-api"]
