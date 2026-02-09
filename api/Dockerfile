# Use a lightweight base image with Python 3.12 preinstalled
FROM python:3.14-slim

# Set /app as the working directory inside the container
WORKDIR /app
# Install uv inside the container (no cache to keep the image small)
RUN pip install --no-cache-dir uv

# Copy dependency metadata first (and lock file if present) to leverage Docker layer caching
COPY pyproject.toml uv.lock* ./
# Install project dependencies:
# - use frozen mode if uv.lock exists
# - otherwise fall back to a normal sync
RUN uv sync --frozen || uv sync

# Copy the FastAPI application source code into the container
COPY app ./app

# Document that the container listens on port 8000
EXPOSE 8000
# Command executed when the container starts:
# - uv run executes within the managed environment
# - uvicorn starts the FastAPI app
# - host 0.0.0.0 makes it accessible from outside the container
# - port 8000 is the API listening port
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]