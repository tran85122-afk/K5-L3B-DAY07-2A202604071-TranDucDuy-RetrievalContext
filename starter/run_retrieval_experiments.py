"""Run Day 7 retrieval & context-budget experiments on a single split (dev first, then locked test)."""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

STARTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STARTER_DIR))

from baseline import hit, pack_context, validate
from retrievers import (
    BM25Retriever,
    GeminiClient,
    GeminiEmbeddingClient,
    GeminiRerankRetriever,
    HybridRRFRetriever,
    PgVectorRetriever,
    TfidfRetriever,
)

RESULTS_DIR = STARTER_DIR / "results"


def resolve_filters(question: dict[str, Any], use_filter: bool) -> dict[str, Any] | None:
    """Use user-provided filter or deterministic chapter pattern from query text (never gold)."""
    if not use_filter:
        return None
    if question.get("filter"):
        return dict(question["filter"])
    m = re.search(r"\bchapter\s+(\d+|overview)\b", question["question"], re.IGNORECASE)
    if m:
        return {"chapter": m.group(1).lower()}
    return None


def reciprocal_rank(marked_results: list[dict[str, Any]]) -> float:
    for idx, item in enumerate(marked_results[:3], start=1):
        if item["hit"]:
            return 1.0 / idx
    return 0.0


def build_pipeline(name: str, chunks: list[dict[str, Any]], embedder: GeminiEmbeddingClient, llm: GeminiClient | None):
    """Return (retriever, top_k, budget_chars, use_filter, description)."""
    if name == "tfidf":
        return TfidfRetriever(chunks), 3, 800, False, "Lexical TF-IDF baseline, top_k=3, budget=800, no filter"
    if name == "bm25":
        return BM25Retriever(chunks), 3, 800, False, "Lexical Okapi BM25 (k1=1.5, b=0.75), top_k=3, budget=800, no filter"
    if name == "dense":
        return PgVectorRetriever(embedder=embedder), 3, 800, False, "Dense pgvector Day 6 (gemini-embedding-001, 768d), top_k=3, budget=800, no filter"
    if name == "hybrid_rrf":
        dense = PgVectorRetriever(embedder=embedder)
        bm25 = BM25Retriever(chunks)
        return HybridRRFRetriever(dense, bm25, candidate_k=20, rrf_k=60), 3, 800, False, "Hybrid RRF (Dense + BM25, k=60), top_k=3, budget=800, no filter"
    if name == "hybrid_rrf_filter_budget1800":
        dense = PgVectorRetriever(embedder=embedder)
        bm25 = BM25Retriever(chunks)
        return HybridRRFRetriever(dense, bm25, candidate_k=20, rrf_k=60), 3, 1800, True, "Hybrid RRF + metadata chapter filter + budget=1800 chars"
    if name == "hybrid_rerank_budget1800":
        if llm is None:
            raise RuntimeError("GeminiClient required for hybrid_rerank_budget1800")
        dense = PgVectorRetriever(embedder=embedder)
        bm25 = BM25Retriever(chunks)
        base = HybridRRFRetriever(dense, bm25, candidate_k=20, rrf_k=60)
        return GeminiRerankRetriever(base, llm=llm, candidate_k=10), 3, 1800, True, "Hybrid RRF top-10 + chapter filter + Gemini-3-Flash-Preview listwise rerank -> top-3, budget=1800"
    raise ValueError(f"Unknown pipeline config: {name}")


def evaluate_pipeline(
    name: str,
    chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    split: str,
    embedder: GeminiEmbeddingClient,
    llm: GeminiClient | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retriever, top_k, budget, use_filter, desc = build_pipeline(name, chunks, embedder, llm)
    embed_calls_0 = embedder.api_calls
    embed_hits_0 = embedder.cache_hits
    llm_calls_0 = llm.api_calls if llm else 0
    llm_ptok_0 = llm.prompt_tokens if llm else 0
    llm_otok_0 = llm.output_tokens if llm else 0

    traces: list[dict[str, Any]] = []
    t_total_start = time.perf_counter()
    for q in questions:
        applied_filter = resolve_filters(q, use_filter)
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=applied_filter, top_k=top_k)
        context, context_ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        marked = [{**r, "hit": hit(r, q["gold"])} for r in results]
        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "strategy": name,
            "filter_applied": applied_filter,
            "top_k": top_k,
            "budget_chars": budget,
            "gold": q["gold"],
            "results": marked,
            "context": context,
            "context_ids": context_ids,
            "context_chars": len(context),
            "elapsed_ms": round(elapsed_ms, 3),
            "hit1": bool(marked and marked[0]["hit"]),
            "hit3": any(r["hit"] for r in marked[:3]),
            "context_hit": any(r["hit"] and r["id"] in context_ids for r in marked),
            "rr3": round(reciprocal_rank(marked), 4),
        })
    total_seconds = time.perf_counter() - t_total_start

    n = len(traces)
    hit1 = sum(t["hit1"] for t in traces)
    hit3 = sum(t["hit3"] for t in traces)
    ctx_hit = sum(t["context_hit"] for t in traces)
    summary = {
        "config": name,
        "description": desc,
        "split": split,
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "use_filter": use_filter,
        "hit1": hit1,
        "hit1_rate": round(hit1 / n, 4),
        "hit3": hit3,
        "hit3_rate": round(hit3 / n, 4),
        "context_hit": ctx_hit,
        "context_hit_rate": round(ctx_hit / n, 4),
        "mrr3": round(statistics.mean(t["rr3"] for t in traces), 4),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
        "mean_ms": round(statistics.mean(t["elapsed_ms"] for t in traces), 2),
        "total_seconds": round(total_seconds, 3),
        "embedding_api_calls": embedder.api_calls - embed_calls_0,
        "embedding_cache_hits": embedder.cache_hits - embed_hits_0,
        "llm_api_calls": (llm.api_calls - llm_calls_0) if llm else 0,
        "llm_prompt_tokens": (llm.prompt_tokens - llm_ptok_0) if llm else 0,
        "llm_output_tokens": (llm.output_tokens - llm_otok_0) if llm else 0,
        "service_cost_usd": 0.0,
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Day 7 retrieval configs on dev or test split.")
    parser.add_argument("--chunks", type=Path, default=STARTER_DIR / "data/converted/chunks.json")
    parser.add_argument("--questions", type=Path, default=STARTER_DIR / "data/converted/questions.json")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "tfidf",
            "bm25",
            "dense",
            "hybrid_rrf",
            "hybrid_rrf_filter_budget1800",
            "hybrid_rerank_budget1800",
        ],
    )
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [q for q in all_questions if q.get("split") == args.split]
    validate(chunks, selected)

    embedder = GeminiEmbeddingClient()
    needs_llm = any("rerank" in c for c in args.configs)
    llm = GeminiClient() if needs_llm else None

    summaries = []
    for cfg in args.configs:
        summary, traces = evaluate_pipeline(cfg, chunks, selected, args.split, embedder, llm)
        out_dir = RESULTS_DIR / f"{args.split}-{cfg}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
        print(
            f"[{args.split}] {cfg:<30} "
            f"Hit@1={summary['hit1']}/{summary['n']} ({summary['hit1_rate']:.3f}) | "
            f"Hit@3={summary['hit3']}/{summary['n']} ({summary['hit3_rate']:.3f}) | "
            f"Ctx={summary['context_hit']}/{summary['n']} ({summary['context_hit_rate']:.3f}) | "
            f"MRR@3={summary['mrr3']:.3f} | med={summary['median_ms']}ms"
        )

    overview_path = RESULTS_DIR / f"{args.split}_retrieval_comparison.json"
    overview_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved comparison to {overview_path}")


if __name__ == "__main__":
    main()
