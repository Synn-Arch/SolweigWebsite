# syntax=docker/dockerfile:1.7
# Two stages: build the React UI, then a GDAL 3.13 runtime with solweig_light.

FROM node:22-alpine AS frontend
WORKDIR /fe
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/frontend/ ./
RUN npm run build


# Official GDAL image: libgdal 3.13.3 + headers, matching the pyproject pin.
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.13.3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    NUMBA_CACHE_DIR=/app/numba-cache \
    SOLWEIG_SCENE_DIR=/app/web/scene \
    SOLWEIG_DATA_DIR=/app/data \
    SOLWEIG_FRONTEND_DIST=/app/web/frontend/dist \
    PORT=8080

# A C++ toolchain is needed once to build the GDAL Python bindings against
# libgdal with the pinned NumPy; it is removed afterwards.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
RUN uv python install 3.11 && uv venv --python 3.11 "$VIRTUAL_ENV"

WORKDIR /app
COPY web/backend/requirements-core.txt web/backend/requirements.txt web/backend/
RUN uv pip install -r web/backend/requirements-core.txt \
    && uv pip install --no-build-isolation "gdal==$(gdal-config --version)" \
    && uv pip install -r web/backend/requirements.txt \
    && python -c "from osgeo import gdal, gdal_array; print('GDAL', gdal.__version__)" \
    && apt-get purge -y build-essential && apt-get autoremove -y

# The numerical package itself (source tree only; tests/docs are excluded).
COPY pyproject.toml LICENSE README.md ./
COPY src/ src/
RUN uv pip install --no-deps . && python -c "import solweig_light; print(solweig_light.__version__)"

COPY web/scene/ web/scene/
COPY web/backend/ web/backend/
COPY --from=frontend /fe/dist web/frontend/dist

# Baseline run at build time: produces the initial overlays and warms the
# Numba cache so the first user-triggered scenario does not pay JIT cost.
ARG SOLWEIG_BUILD_THREADS=2
RUN mkdir -p "$NUMBA_CACHE_DIR" "$SOLWEIG_DATA_DIR" \
    && cd web/backend \
    && SOLWEIG_THREADS=$SOLWEIG_BUILD_THREADS python -m app.precompute

WORKDIR /app/web/backend
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
