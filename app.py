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

@app.post("/query")
def query(request: QueryRequest):
    question = request.question
    bypass = request.bypass_cache

    # ── Check Cache First ────────────────────────────────────────────────────
    if not bypass:
        cached = api_cache.get(question)
        if cached is not None:
            obs.log(
                request_id=str(uuid.uuid4()),
                question=question,
                latency={
                    "IG_latency": 0.0,
                    "IC_latency": 0.0,
                    "Response_latency": 0.0,
                    "Total_latency": 0.0
                },
                retrieval_metrics={
                    "retrieval_latency": 0.0,
                    "no_documents": 0,
                    "best_score": 0.0,
                    "avg_score": 0.0,
                    "confidence_score_pass": False
                },
                memory_usage=obs.get_memory_usage(),
                blocked_input=cached.get("blocked", False),
                cache_hit=True
            )
            return cached

    # Cache miss or bypass: run standard process
    main.init_metrics()
    blocked_input, intent, ig_response, raw_intent = main.run_parallel_checks(question)
    
    category = "BLOCKED"
    answer = ""

    if blocked_input:
        category = "BLOCKED"
        answer = "I cannot assist with that request due to safety policies. Please contact official support."
    else:
        category    = intent.get("category", "OUT_OF_DOMAIN").strip().upper()
        instruction = intent.get("instruction", "")

        if category == "JAILBREAK":
            answer = (
                "I'm sorry, but I'm unable to assist with that request. "
                "My behaviour cannot be overridden or modified. "
                "If you have a genuine insurance query, I'm happy to help."
            )

        elif category == "OUT_OF_DOMAIN":
            answer = (
                "I'm an Insurance Support Assistant and can only help with "
                "insurance-related queries. Your question appears to be outside "
                "my area of expertise. Please contact the relevant service for assistance."
            )

        elif category == "INSURANCE":
            agent_response = mainAgent(question, instruction=instruction, use_rag=True)
            answer = validate_output(agent_response)

        elif category == "AMBIGUOUS":
            agent_response = mainAgent(question, instruction=instruction, use_rag=False)
            answer = validate_output(agent_response)

        elif category == "FRAUD":
            agent_response = mainAgent(question, instruction=instruction, use_rag=True)
            answer = validate_output(agent_response)

        else:
            answer = (
                "I'm unable to process your request at this time. "
                "Please contact our support team for assistance."
            )

    # ── Step 3: Observability logging ─────────────────────────────────────────
    metrics = main.get_metrics()
    obs.log(
        request_id=str(uuid.uuid4()),
        question=question,
        latency={
            "IG_latency": metrics["IG_latency"],
            "IC_latency": metrics["IC_latency"],
            "Response_latency": metrics["request_latency"],
            "Total_latency": (
                metrics["checks_latency"]
                + metrics["retrieval_latency"]
                + metrics["request_latency"]
            )
        },
        retrieval_metrics={
            "retrieval_latency": metrics["retrieval_latency"],
            "no_documents": metrics["retrieval_metrics_data"]["no_documents"],
            "best_score": metrics["retrieval_metrics_data"]["best_score"],
            "avg_score": metrics["retrieval_metrics_data"]["avg_score"],
            "confidence_score_pass": (
                metrics["retrieval_metrics_data"]["best_score"] >= main.CONFIDENCE_THRESHOLD
            ),
        },
        memory_usage={
            "cpu_percent": metrics["memory_usage"]["cpu_percent"],
            "ram_percent": metrics["memory_usage"]["ram_percent"],
            "ram_used_mb": metrics["memory_usage"]["ram_used_mb"]
        },
        blocked_input=blocked_input,
        cache_hit=False
    )

    response_data = {
        "answer": answer,
        "blocked": blocked_input,
        "category": category
    }

    # ── Step 4: Write to Cache ───────────────────────────────────────────────
    api_cache.set(question, response_data)

    return response_data
