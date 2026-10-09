# EquiForge — deterministic algorithmic-fairness toolkit.
FROM python:3.13-slim

WORKDIR /app

# Install runtime dependencies first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the toolkit.
COPY . .
RUN pip install --no-cache-dir pytest ruff

# Default: run the full benchmark demo (also the CI smoke test).
CMD ["python", "examples/run_demo.py"]
