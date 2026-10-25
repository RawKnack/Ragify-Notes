"""
evaluation/benchmark.py — RAGAS benchmark with incremental evaluation caching.

Runs retrieval modes against a test question set, evaluates
with RAGAS metrics per question, caches results to disk,
and logs summary results to MLflow for comparison.

Features:
- Incremental Caching: Saves per-question evaluation scores immediately to disk.
  If an API rate limit occurs or script is interrupted, re-running
  instantly resumes from where it left off!

Usage:
    python -m evaluation.benchmark
    python -m evaluation.benchmark --modes naive hybrid hybrid_rerank
    python -m evaluation.benchmark --test-file evaluation/test_questions.json
    python -m evaluation.benchmark --limit 2          # run only 2 questions, then stop
    python -m evaluation.benchmark --limit 2 --modes hybrid_rerank  # combine flags

View results:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
    Then open http://localhost:5001
"""

import os
import sys
import json
import argparse
import warnings
import logging
import time
from datetime import datetime

# Suppress Google GenAI SDK Automatic Function Calling (AFC) & deprecation warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.genai")
logging.getLogger("google").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

# Set UTF-8 output encoding for Windows terminal compatibility
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Allow file store for MLflow tracking
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from datasets import Dataset

# Compatibility fix for newer LangChain version imports in RAGAS
try:
    import langchain_google_vertexai
    sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai
except ImportError:
    pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall

from src.retriever import query as rag_query
from config.settings import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    EMBED_MODEL_NAME,
    RERANK_MODEL_NAME,
    RRF_K,
    DEFAULT_ALPHA,
    GEMINI_MODEL,
    GEMINI_API_KEY,
)

EVAL_CACHE_FILE = "evaluation/results/eval_cache.json"

# Initialize Gemini LLM with LangChain OutputParser backend & local embeddings
evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
    )
)
evaluator_embed = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME))

METRICS = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embed, strictness=1),
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
]


def load_eval_cache() -> dict:
    """Loads cached question evaluation scores from disk."""
    if os.path.exists(EVAL_CACHE_FILE):
        try:
            with open(EVAL_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_eval_cache(cache: dict):
    """Saves evaluation cache to disk immediately."""
    os.makedirs(os.path.dirname(EVAL_CACHE_FILE), exist_ok=True)
    with open(EVAL_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def load_test_questions(filepath: str) -> list[dict]:
    """Load test questions with ground truth."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single_question(
    q: dict,
    mode: str,
    cache: dict,
) -> tuple[dict, dict]:
    """
    Evaluates a single question using the specified RAG mode.
    Uses disk cache to skip already-evaluated questions.

    Returns:
        (pipeline_data, scores_dict)
    """
    cache_key = f"{mode}::{q['question']}"

    if cache_key in cache:
        print(" ⚡ [Cached]")
        cached_entry = cache[cache_key]
        return cached_entry["pipeline_data"], cached_entry["scores"]

    # Step 1: Run RAG Pipeline
    result = rag_query(
        question=q["question"],
        top_k=5,
        filter_confidence=None,
        mode=mode,
        verbose=False,
    )

    contexts = [s["text"] for s in result.get("sources", [])]
    pipeline_data = {
        "question": q["question"],
        "answer": result.get("answer", ""),
        "contexts": contexts,
        "ground_truth": q.get("ground_truth", ""),
    }

    # Step 2: Evaluate Single Question with RAGAS
    dataset = Dataset.from_dict({
        "question": [pipeline_data["question"]],
        "answer": [pipeline_data["answer"]],
        "contexts": [pipeline_data["contexts"]],
        "ground_truth": [pipeline_data["ground_truth"]],
    })

    try:
        ragas_result = evaluate(
            dataset,
            metrics=METRICS,
            llm=evaluator_llm,
            embeddings=evaluator_embed,
            run_config=RunConfig(max_workers=1, max_retries=3, max_wait=15, timeout=180),
        )

        # Extract metric scores
        df = ragas_result.to_pandas()
        scores = {}
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if col in df.columns:
                val = df[col].iloc[0]
                scores[col] = float(val) if (val is not None and not pd_isna(val)) else 0.0
            else:
                scores[col] = 0.0

        print(f" ✓ F={scores.get('faithfulness',0):.2f} R={scores.get('answer_relevancy',0):.2f} P={scores.get('context_precision',0):.2f} C={scores.get('context_recall',0):.2f}")

    except Exception as e:
        root_cause = getattr(e, '__cause__', None) or getattr(e, '__context__', None) or e
        print(f"\n ❌ Eval error ({type(root_cause).__name__}): {root_cause}")
        scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}

    # Save to cache whenever valid metrics are calculated
    has_valid_scores = any(v > 0.0 for v in scores.values())
    if has_valid_scores:
        cache[cache_key] = {
            "pipeline_data": pipeline_data,
            "scores": scores,
            "timestamp": datetime.now().isoformat(),
        }
        save_eval_cache(cache)
    else:
        print(" ⚠️  No valid metrics returned — will retry on next run.")

    return pipeline_data, scores


def pd_isna(val) -> bool:
    """Helper to check for NaN / None values safely."""
    try:
        import math
        return val is None or math.isnan(float(val))
    except Exception:
        return False


def benchmark(
    test_file: str = "evaluation/test_questions.json",
    modes: list[str] = None,
    limit: int = None,
):
    """
    Run RAGAS benchmark across specified retrieval modes with incremental caching.
    """
    if modes is None:
        modes = ["naive", "hybrid", "hybrid_rerank"]

    print(f"\n📂 Loading test questions from {test_file}...")
    questions = load_test_questions(test_file)
    print(f"   Found {len(questions)} test questions")

    if limit is not None:
        # Only pick questions not yet cached for each mode, up to `limit`
        cache_check = load_eval_cache()
        uncached = [q for q in questions if not all(
            f"{m}::{q['question']}" in cache_check for m in modes
        )]
        questions = uncached[:limit]
        print(f"   ⚡ --limit {limit}: running {len(questions)} uncached question(s) this session")

    cache = load_eval_cache()

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_mode_results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"🔬 BENCHMARKING MODE: {mode.upper()}")
        print(f"{'='*60}")

        mode_pipeline_results = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        mode_scores_list = []

        with mlflow.start_run(run_name=f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("retrieval_mode", mode)
            mlflow.log_param("embedding_model", EMBED_MODEL_NAME)
            mlflow.log_param("answer_model", GEMINI_MODEL)
            mlflow.log_param("rerank_model", RERANK_MODEL_NAME if mode == "hybrid_rerank" else "none")
            mlflow.log_param("rrf_k", RRF_K)
            mlflow.log_param("alpha", DEFAULT_ALPHA)
            mlflow.log_param("top_k", 5)
            mlflow.log_param("num_questions", len(questions))

            for i, q in enumerate(questions):
                print(f"  [{i+1}/{len(questions)}] {q['question']} (mode={mode})...", end="")
                p_data, scores = evaluate_single_question(q, mode, cache)

                mode_pipeline_results["question"].append(p_data["question"])
                mode_pipeline_results["answer"].append(p_data["answer"])
                mode_pipeline_results["contexts"].append(p_data["contexts"])
                mode_pipeline_results["ground_truth"].append(p_data["ground_truth"])
                mode_scores_list.append(scores)

            # Compute mean scores for this mode
            metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            mean_scores = {}
            for k in metric_keys:
                vals = [s[k] for s in mode_scores_list if k in s]
                mean_scores[k] = float(sum(vals) / len(vals)) if vals else 0.0

            print(f"\n  📊 Mean Scores for {mode.upper()}:")
            for metric_name, score in mean_scores.items():
                mlflow.log_metric(metric_name, float(score))
                print(f"    {metric_name}: {score:.4f}")

            all_mode_results[mode] = mean_scores

            # Save raw pipeline results as artifact
            results_path = f"evaluation/results/{mode}_results.json"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(mode_pipeline_results, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(results_path)

    # Print comparison summary
    print(f"\n{'='*60}")
    print("📊 BENCHMARK COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Mode':<20} {'Faithfulness':<15} {'Relevancy':<15} {'Precision':<15} {'Recall':<15}")
    print("─" * 80)

    for mode, scores in all_mode_results.items():
        if "error" in scores:
            print(f"{mode:<20} {'ERROR':<15}")
        else:
            f = scores.get("faithfulness", 0)
            r = scores.get("answer_relevancy", 0)
            p = scores.get("context_precision", 0)
            c = scores.get("context_recall", 0)
            print(f"{mode:<20} {f:<15.4f} {r:<15.4f} {p:<15.4f} {c:<15.4f}")

    print(f"\n✅ Results logged to MLflow. View with: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")

    return all_mode_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS Benchmark with Incremental Caching")
    parser.add_argument("--test-file", type=str, default="evaluation/test_questions.json",
                        help="Path to test questions JSON file")
    parser.add_argument("--modes", nargs="+", default=["naive", "hybrid", "hybrid_rerank"],
                        choices=["naive", "hybrid", "hybrid_rerank"],
                        help="Retrieval modes to benchmark")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of uncached questions to evaluate per run (for rate-limit management)")
    args = parser.parse_args()

    benchmark(test_file=args.test_file, modes=args.modes, limit=args.limit)
