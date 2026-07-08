# Insurance Customer Support Agent — Project Documentation

> A Retrieval-Augmented Generation (RAG) customer support chatbot for insurance queries, containerized with Docker and served via a FastAPI REST API. Uses a local Ollama LLM (`gemma2:2b`) for all inference, with a multi-stage safety pipeline.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Pipeline Flow](#pipeline-flow)
4. [Source Files](#source-files)
5. [API Reference](#api-reference)
6. [Observability & Logs](#observability--logs)
7. [Dockerfile](#dockerfile)
8. [Requirements](#requirements)
9. [Running the Application](#running-the-application)
10. [Latency Optimizations](#latency-optimizations)

---

## Architecture Overview

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI (app.py)                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               In-Memory Cache (cache.py)            │   │
│  │  SHA-256 hash of normalized query → cached response │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │ Cache Miss                            │
│                     ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Safety Pipeline (main.py)                │   │
│  │                                                     │   │
│  │  validate_input()  ←── Regex Guardrail (fast)       │   │
│  │        │                                            │   │
│  │  [Pattern Blocked?]                                 │   │
│  │   Yes → run_guardrail_and_intent_parallel()         │   │
│  │         ├── InputGuardrail (LLM)   ┐ parallel       │   │
│  │         └── ClassifyIntent (LLM)   ┘ threads        │   │
│  │   No  → ClassifyIntent (LLM) only                   │   │
│  │                                                     │   │
│  │  Intent → JAILBREAK / FRAUD / OUT_OF_DOMAIN /       │   │
│  │           INSURANCE / AMBIGUOUS / BLOCKED           │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                       │
│                     ▼ (INSURANCE, FRAUD, AMBIGUOUS only)   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RAG Pipeline (vector.py)               │   │
│  │  Query → ChromaDB (nomic-embed-text embeddings)     │   │
│  │       → Top-3 relevant insurance FAQ docs           │   │
│  │       → Confidence threshold check (≥ 0.35)         │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                       │
│                     ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Main Agent (mainAgent in main.py)        │   │
│  │  gemma2:2b + RAG context + role prompt              │   │
│  │  → validate_output() → redact_pii() → answer        │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                       │
│                     ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Observability Log (observability.py)               │   │
│  │  → logs/logs_insur.jsonl (or logs/logs.jsonl)       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
JSON Response: { answer, blocked, category }
```

---

## Project Structure

```
summer-project/
├── app.py               # FastAPI server — REST API layer
├── main.py              # Core pipeline: guardrails, intent classifier, main agent
├── vector.py            # Vector DB setup and RAG retrieval (ChromaDB + Ollama embeddings)
├── cache.py             # Thread-safe in-memory response cache
├── observability.py     # Logging, latency, and memory metrics
├── insurance.csv        # Primary knowledge base (1000 Q&A pairs)
├── company_faq.csv      # Secondary knowledge base (company FAQs)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Multi-stage Docker build
├── .dockerignore        # Excludes .venv, .cache, db files, etc.
└── logs/
    ├── logs_insur.jsonl # Query logs for insurance.csv dataset
    └── logs.jsonl       # Query logs for company_faq.csv dataset
```

---

## Pipeline Flow

Every incoming query passes through this ordered pipeline:

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **Cache Lookup** | Normalized SHA-256 hash checked. Returns instantly if matched. |
| 2 | **Regex Guardrail** (`validate_input`) | Fast pattern match for injection keywords, jailbreak phrases. |
| 3a | **LLM Input Guardrail** (`InputGuardrail`) | Only runs if Step 2 flags the input. Runs parallel with Step 3b. |
| 3b | **Intent Classifier** (`ClassifyIntent`) | LLM classifies the query into one of 5 intent categories. |
| 4 | **RAG Retrieval** (`retrieve_with_confidence`) | Vector search for top-3 relevant docs; confidence threshold: 0.35. |
| 5 | **Main Agent** (`mainAgent`) | LLM generates the answer using the retrieved context + system prompt. |
| 6 | **Output Guardrail** (`validate_output`) | Blocks responses containing sensitive info; applies PII redaction. |
| 7 | **Observability Log** | Writes full trace (latency, retrieval scores, memory usage) to JSONL. |
| 8 | **Cache Write** | Stores response in memory for future identical queries. |

---

## Source Files

### app.py

FastAPI application. Entry point for all API requests.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "healthy"}` |
| `POST` | `/query` | Submit a question, returns answer + metadata |
| `GET` | `/docs` | Swagger UI (auto-generated) |

**Request body (POST /query):**
```json
{
  "question": "Can I cancel my insurance policy?",
  "bypass_cache": false
}
```

**Response:**
```json
{
  "answer": "Most policies can be cancelled subject to the insurer's terms...",
  "blocked": false,
  "category": "INSURANCE"
}
```

```python
# app.py — Full Source
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import main
from main import validate_input, InputGuardrail, ClassifyIntent, parse_intent, mainAgent, validate_output
import observability as obs
from cache import APICache

class QueryRequest(BaseModel):
    question: str
    bypass_cache: bool = False

app = FastAPI(title="Customer Support Agent")
api_cache = APICache()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/query")
def query(request: QueryRequest):
    question = request.question
    bypass = request.bypass_cache

    # Check Cache First
    if not bypass:
        cached = api_cache.get(question)
        if cached is not None:
            obs.log(request_id=str(uuid.uuid4()), question=question,
                    latency={"IG_latency":0.0,"IC_latency":0.0,"Response_latency":0.0,"Total_latency":0.0},
                    retrieval_metrics={"retrieval_latency":0.0,"no_documents":0,"best_score":0.0,"avg_score":0.0,"confidence_score_pass":False},
                    memory_usage=obs.get_memory_usage(), blocked_input=cached.get("blocked", False), cache_hit=True)
            return cached

    # Cache miss: run full pipeline
    main.init_metrics()
    blocked_input, intent, ig_response, raw_intent = main.run_parallel_checks(question)
    category = "BLOCKED"
    answer = ""

    if blocked_input:
        answer = "I cannot assist with that request due to safety policies. Please contact official support."
    else:
        category = intent.get("category", "OUT_OF_DOMAIN").strip().upper()
        instruction = intent.get("instruction", "")

        if   category == "JAILBREAK":    answer = "I'm sorry, but I'm unable to assist with that request. My behaviour cannot be overridden or modified. If you have a genuine insurance query, I'm happy to help."
        elif category == "OUT_OF_DOMAIN": answer = "I'm an Insurance Support Assistant and can only help with insurance-related queries. Your question appears to be outside my area of expertise."
        elif category == "INSURANCE":    answer = validate_output(mainAgent(question, instruction=instruction, use_rag=True))
        elif category == "AMBIGUOUS":    answer = validate_output(mainAgent(question, instruction=instruction, use_rag=False))
        elif category == "FRAUD":        answer = validate_output(mainAgent(question, instruction=instruction, use_rag=True))
        else: answer = "I'm unable to process your request at this time. Please contact our support team."

    # Log + Cache + Return
    metrics = main.get_metrics()
    obs.log(request_id=str(uuid.uuid4()), question=question,
            latency={"IG_latency":metrics["IG_latency"],"IC_latency":metrics["IC_latency"],"Response_latency":metrics["request_latency"],"Total_latency":metrics["checks_latency"]+metrics["retrieval_latency"]+metrics["request_latency"]},
            retrieval_metrics={"retrieval_latency":metrics["retrieval_latency"],"no_documents":metrics["retrieval_metrics_data"]["no_documents"],"best_score":metrics["retrieval_metrics_data"]["best_score"],"avg_score":metrics["retrieval_metrics_data"]["avg_score"],"confidence_score_pass":metrics["retrieval_metrics_data"]["best_score"]>=main.CONFIDENCE_THRESHOLD},
            memory_usage={"cpu_percent":metrics["memory_usage"]["cpu_percent"],"ram_percent":metrics["memory_usage"]["ram_percent"],"ram_used_mb":metrics["memory_usage"]["ram_used_mb"]},
            blocked_input=blocked_input, cache_hit=False)
    response_data = {"answer": answer, "blocked": blocked_input, "category": category}
    api_cache.set(question, response_data)
    return response_data
```

---

### main.py

Core business logic. Implements the full safety and inference pipeline.

**Model Configuration:**

```python
# Guardrail model — for InputGuardrail and IntentClassifier
guardrail_model = OllamaLLM(model="gemma2:2b", keep_alive=300, num_ctx=1024, temperature=0.0, num_predict=60, base_url="http://localhost:11434")

# Main model — for the agent response
main_model = OllamaLLM(model="gemma2:2b", keep_alive=300, num_ctx=768, temperature=0.2, num_predict=100, base_url="http://localhost:11434")

CONFIDENCE_THRESHOLD = 0.35
```

**Intent Categories:**

| Category | Trigger | Response Strategy |
|----------|---------|-------------------|
| `JAILBREAK` | Tries to bypass instructions | Hard refusal, no LLM call |
| `FRAUD` | Asks to lie/fake/exaggerate claims | LLM analyses risk; refuses if high |
| `OUT_OF_DOMAIN` | Unrelated to insurance | Polite redirect, no LLM call |
| `INSURANCE` | Clear, specific insurance query | RAG retrieval + LLM answer |
| `AMBIGUOUS` | Insurance-related but too vague | LLM asks clarifying questions (no RAG) |
| `BLOCKED` | Regex guardrail hard-block | Safety policy response |

**Blocked Input Patterns (regex):**
- Prompt injection: `ignore all previous instructions`, `reveal system prompt`, `system instructions`
- Credential requests: `password`, `api key`, `secret`, `token`, `credential`
- Jailbreak: `bypass safety`, `unrestricted ai`, `developer mode`, `override`, `pretend you are`

**PII Redaction:**
- Emails → `[customer_email]` (company email preserved)
- Phone numbers → `[customer_phone]` (company phone preserved)

---

### vector.py

Sets up ChromaDB and provides RAG retrieval.

```python
CSV_FILE = "insurance.csv"       # Controls which dataset is loaded
df = pd.read_csv(CSV_FILE)       # 1000 rows: id, question, answer, category

# Embeddings: nomic-embed-text via Ollama, cached to .cache/embeddings/
# Vector DB: ChromaDB persisted at chrome_langchain_db/
# Retrieval: Top-3 results with cosine similarity scores
# Auto-rebuild: if ChromaDB count != CSV row count
```

---

### cache.py

Thread-safe in-memory response cache. Lifetime = server process lifetime.

```python
class APICache:
    # SHA-256 hash of lowercase+normalized query → dict response
    def get(query: str) -> dict | None   # None on miss
    def set(query: str, response: dict)  # Deep copy stored
    def clear()                           # Empties cache
```

---

### observability.py

Collects and writes per-request metrics as JSONL.

**Log file routing:**
- Inspects `vector.py` at runtime to detect loaded CSV
- `insurance.csv` → `logs/logs_insur.jsonl`
- Other CSV → `logs/logs.jsonl`
- Uses `os.path.abspath(__file__)` for reliable path resolution inside Docker

**Metrics tracked:** timestamp, request_id, question, IG/IC/Response/Total latency, retrieval latency, document count, relevance scores, confidence pass, CPU %, RAM %, RAM MB, blocked_input, cache_hit.

---

## API Reference

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Does travel insurance cover lost baggage?", "bypass_cache": false}'
```

Response:
```json
{"answer": "Travel insurance typically covers lost baggage...", "blocked": false, "category": "INSURANCE"}
```

### GET /health

```bash
curl http://localhost:8000/health
# Response: {"status": "healthy"}
```

### Swagger UI

Open `http://localhost:8000/docs` in your browser.

---

## Observability & Logs

Log files are JSONL format (one JSON object per line) in the `logs/` directory.

**Sample log entry (cache miss):**
```json
{
  "timestamp": "2026-07-08T13:33:41.858670",
  "request_id": "b87acc54-...",
  "question": "I lost my baggage at the airport. Does travel insurance cover lost baggage?",
  "latency": {"IG Latency": 0.0, "IC Latency": 10.156, "Response Latency": 9.199, "Total Latency": 19.42},
  "retrieval_metrics": {"Retrieval Latency": 0.065, "No. of Documents": 3, "Best Relevance Score": 0.671, "Avg Relevance Score": 0.658, "Passes Confidence Scores": true},
  "memory_usage": {"cpu_percent": 0.3, "ram_percent": 2.58, "ram_used_mb": 201.27},
  "blocked_input": false,
  "cache_hit": false
}
```

**Cache hit entries** always show 0ms latency for all fields.

---

## Dockerfile

```dockerfile
# Stage 1: Pull and bake Ollama models
FROM ollama/ollama:latest AS model-builder
RUN ollama serve & until ollama list >/dev/null 2>&1; do sleep 1; done && \
    ollama pull gemma2:2b && ollama pull nomic-embed-text:latest
RUN mkdir -p /usr/lib/ollama-cpu && \
    cp -d /usr/lib/ollama/llama-server /usr/lib/ollama/llama-quantize /usr/lib/ollama/*.so* /usr/lib/ollama-cpu/

# Stage 2: Final runtime image
FROM python:3.11-slim
ENV OLLAMA_DEBUG=WARN PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=model-builder /usr/bin/ollama /usr/bin/ollama
COPY --from=model-builder /usr/lib/ollama-cpu /usr/lib/ollama
COPY --from=model-builder /root/.ollama /root/.ollama
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8000
CMD ["sh", "-c", "ollama serve & until curl -s http://localhost:11434/api/tags >/dev/null; do sleep 0.5; done && uvicorn app:app --host 0.0.0.0 --port 8000"]
```

---

## Requirements

```
langchain
langchain_ollama
langchain_classic
langchain_core
langchain_chroma
pandas
psutil
fastapi
uvicorn
```

---

## Running the Application

### Docker (recommended)

```bash
# Build
docker build -t summer-project .

# Run with log persistence (PowerShell)
docker run -p 8000:8000 -v "${PWD}/logs:/app/logs" summer-project

# With NVIDIA GPU acceleration
docker run -p 8000:8000 -v "${PWD}/logs:/app/logs" --gpus all summer-project

# Stop all containers
docker stop $(docker ps -q)
```

**Access:** `http://localhost:8000/docs`

### Local (dev)

```bash
.venv\Scripts\activate       # Windows
ollama serve                 # Separate terminal
uvicorn app:app --port 8000 --reload
```

---

## Latency Optimizations

**Typical latency breakdown (CPU inference):**

| Component | Latency |
|-----------|---------|
| Cache hit | ~0ms |
| Intent Classifier | 4–10s (main bottleneck) |
| Main Agent | 2–10s |
| RAG Retrieval | ~0.07s |
| Regex Guardrail | ~0ms |

**Applied fixes:**

| Fix | Change | Impact |
|-----|--------|--------|
| `num_predict=60` on guardrail_model | Caps IC output tokens | Prevents runaway generation |
| Shortened IC prompt | ~800 → ~250 tokens | ~30-40% faster prompt processing |
| `num_ctx=768` on main_model | Reduced from 1024 | Lower attention computation cost |
| In-memory cache | SHA-256 deduplication | Repeat queries: 0ms |

**Future improvements:**

| Option | Description | Speedup |
|--------|-------------|---------|
| GPU passthrough | `--gpus all` with `nvidia-container-toolkit` | 5–10x |
| Quantized model | Switch to `gemma2:2b-q4` | 2–3x on CPU |
| FastAPI streaming | `StreamingResponse` | Reduced perceived latency |
