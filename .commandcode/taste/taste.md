# Taste

- Prefers `uv` for Python dependency management and Docker installs (e.g. `uv sync --locked` in builds) over pip/poetry. Confidence: 0.7
- Prefers Debian "slim trixie" base images for Python containers (`python:<ver>-slim-trixie`). Confidence: 0.6
- Wants multi-stage Dockerfiles (separate builder/runtime stages) rather than single-stage builds. Confidence: 0.6
