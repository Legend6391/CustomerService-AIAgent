#Refer to CGPT 2nd query for observability
#latency
#accuracy(relevance_scores)
#CPU/RAM usage

import json
import os
import psutil
from datetime import datetime

# Initialize process global reference for CPU tracking
_process = psutil.Process(os.getpid())
# Call cpu_percent once at startup to initialize the baseline tracker
_process.cpu_percent(interval=None)

def get_memory_usage():
    mem_info = _process.memory_info()
    ram_used_mb = round(mem_info.rss / (1024 * 1024), 2)
    
    total_memory = psutil.virtual_memory().total
    ram_percent = round((mem_info.rss / total_memory) * 100, 2)
    
    cpu_usage = _process.cpu_percent(interval=None)
    
    return {
        "cpu_percent": cpu_usage,
        "ram_percent": ram_percent,
        "ram_used_mb": ram_used_mb
    }

def retrieval_metrics(scores):

    if not scores:
        return {
            "no_documents": 0,
            "best_score": 0,
            "avg_score": 0
        }

    return {
        "no_documents": len(scores),
        "best_score": round(max(scores), 3),
        "avg_score": round(sum(scores) / len(scores), 3)
    }

# log.py
# metadata of retrieved documents, relevance scores, and response/retrieval times for each query to analyze the performance of 
# the retrieval system and identify areas for improvement.

def log(
    request_id,
    question,
    latency,
    retrieval_metrics,
    memory_usage,
    blocked_input,
    cache_hit=False
):

    trace = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "question": question,
        "latency": {
            "IG Latency": round(latency["IG_latency"], 3),
            "IC Latency": round(latency.get("IC_latency", 0.0), 3),
            "Response Latency": round(latency["Response_latency"], 3),
            "Total Latency": round(latency["Total_latency"], 3)
        },
        "retrieval_metrics": {
            "Retrieval Latency": round(retrieval_metrics["retrieval_latency"], 3),
            "No. of Documents": retrieval_metrics["no_documents"],
            "Best Relevance Score": round(retrieval_metrics["best_score"], 3),
            "Avg Relevance Score": round(retrieval_metrics["avg_score"], 3),
            "Passes Confidence Scores": retrieval_metrics["confidence_score_pass"]
        },
        "memory_usage": {
            "cpu_percent": round(memory_usage["cpu_percent"], 3),
            "ram_percent": round(memory_usage["ram_percent"], 3),
            "ram_used_mb": round(memory_usage["ram_used_mb"], 3)
        },
        "blocked_input": blocked_input,
        "cache_hit": cache_hit
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = "logs.jsonl"
    try:
        # Inspect vector.py to see which CSV is configured for loading
        vector_path = os.path.join(current_dir, "vector.py")
        with open(vector_path, "r", encoding="utf-8") as vf:
            vector_code = vf.read()
        if "insurance.csv" in vector_code:
            log_file = "logs_insur.jsonl"
    except Exception:
        pass

    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")
    
    return trace