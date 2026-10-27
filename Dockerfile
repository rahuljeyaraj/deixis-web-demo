# The judges' web demo as one image (web demo item 17): the page, built; the server; the deixis
# core; Parakeet. Built ON the rented server from the source copied there (deploy/run.sh), so
# nothing large is uploaded from home. The Nebius key is never in the image: it comes from a
# root-only file on the server at run time (deploy/compose.yaml).
#
#   docker build -t deixis-webdemo .                 # the core from wheels/ if there, else PyPI
#
# The core: a wheel in wheels/ (deploy/run.sh builds it from ../deixis) wins; without one the
# image installs deixis==0.1.0 from PyPI (once it's published, about Thu 15 Oct).

# ---- the page
FROM node:22-slim AS page
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
# prebuild packs the arm model
RUN npm run build

# ---- the server
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY wheels/ /wheels/
RUN set -e; \
    if ls /wheels/deixis-*.whl >/dev/null 2>&1; then \
      pip install "$(ls /wheels/deixis-*.whl | head -n 1)[mcp]"; \
    else \
      pip install "deixis[mcp]==0.1.0"; \
    fi
COPY pyproject.toml README.md LICENSE NOTICE.md LIMITATIONS.md ./
COPY server/ server/
COPY profiles/ profiles/
COPY scripts/fetch_models.py scripts/
# deixis is already installed above; the project's own sources point it at ../deixis
RUN pip install -e . --no-deps \
 && pip install "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.9" \
      "pydantic-settings>=2.3" "numpy>=1.26" "openai>=1.40" "sherpa-onnx>=1.12" "onnxruntime>=1.18"
# Parakeet (about 630 MB), downloaded here, on the server
RUN python scripts/fetch_models.py --dir /app/.cache/models
COPY --from=page /web/dist /app/web/dist

# Tuned for 2 vCPU (was 2 workers x 4 threads on rahul's PC): check M31 on the first runs.
ENV HOST=0.0.0.0 PORT=8000 \
    WEB_DIST=/app/web/dist ASR_MODELS_DIR=/app/.cache/models ASR_WORKERS=1 ASR_THREADS=2 \
    PROXY_HOPS=1 COMMAND_LOG=/data/logs/commands.jsonl \
    DEIXIS_MODE=mcp DEIXIS_FALLBACK=direct
RUN useradd --create-home --uid 10001 demo && mkdir -p /data/logs && chown -R demo /data
USER demo
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=8).status == 200 else 1)"
CMD ["webdemo"]
