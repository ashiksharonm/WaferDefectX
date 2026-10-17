# syntax=docker/dockerfile:1
# Stage 1: install dependencies and train the model on synthetic wafers (about a minute on CPU).
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY wafer_inspect/ wafer_inspect/
RUN python -m wafer_inspect.train --out models

# Stage 2: slim non-root runtime (uid 1000, as Hugging Face Spaces expects).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app PORT=7860 PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser /app/models models/
COPY --chown=appuser wafer_inspect/ wafer_inspect/
COPY --chown=appuser samples/ samples/
RUN mkdir -p data && chown appuser data
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','7860'), timeout=3)"
CMD ["sh", "-c", "exec uvicorn wafer_inspect.api:app --host 0.0.0.0 --port ${PORT}"]
