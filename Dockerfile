FROM python:3.12-slim

# libgomp is required by XGBoost's OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY voltguard ./voltguard
COPY config ./config
COPY api ./api
COPY scripts ./scripts
# Editable install keeps PROJECT_ROOT at /app, where models/ and data/ live.
RUN pip install -e .

COPY models ./models

RUN useradd --create-home --uid 1000 voltguard && chown -R voltguard /app
USER voltguard

ENV DATABASE_URL=sqlite:////app/data/voltguard.db
RUN mkdir -p /app/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
