# ==========================================
# STAGE 1: Download Ollama Models
# ==========================================
FROM ollama/ollama:latest AS model-builder

# Start Ollama server in background, poll until responsive, and pull models
RUN ollama serve & \
    until ollama list >/dev/null 2>&1; do echo "Waiting for Ollama..." && sleep 1; done && \
    ollama pull gemma2:2b && \
    ollama pull nomic-embed-text:latest

# Extract only the CPU-only llama-server binary and libraries (omitting ~4GB CUDA/ROCm GPU runtimes)
RUN mkdir -p /usr/lib/ollama-cpu && \
    cp -d /usr/lib/ollama/llama-server /usr/lib/ollama/llama-quantize /usr/lib/ollama/*.so* /usr/lib/ollama-cpu/

# ==========================================
# STAGE 2: Final Runtime Image
# ==========================================
FROM python:3.11-slim
ENV OLLAMA_DEBUG=WARN

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy Ollama binary, CPU-only libraries, and the downloaded models from the builder stage
COPY --from=model-builder /usr/bin/ollama /usr/bin/ollama
COPY --from=model-builder /usr/lib/ollama-cpu /usr/lib/ollama
COPY --from=model-builder /root/.ollama /root/.ollama

# Install minimal runtime dependencies (curl is useful for container health checks)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (no build-essential needed; pre-built wheels exist for all dependencies)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (utilizes .dockerignore to exclude local .venv, .cache, and db files)
COPY . /app

# Expose port: 8000 for FastAPI application
EXPOSE 8000

# Start Ollama in the background, wait for it to be ready, then start the FastAPI app
CMD ["sh", "-c", "ollama serve & until curl -s http://localhost:11434/api/tags >/dev/null; do sleep 0.5; done && uvicorn app:app --host 0.0.0.0 --port 8000"]
