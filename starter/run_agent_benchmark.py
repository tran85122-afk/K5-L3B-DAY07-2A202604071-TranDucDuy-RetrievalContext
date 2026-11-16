"""Run the real Agentic RAG pipeline on dev or test cases and save full traces."""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

STARTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STARTER_DIR))

from agent_adapters import GeminiController
from agentic import run_agent
from retrievers import (
    BM25Retriever,
    GeminiClient,
    GeminiEmbeddingClient,
    GeminiRerankRetriever,
    HybridRRFRetriever,
    PgVectorRetriever,
    TfidfRetriever,
)
from run_retrieval_experiments import resolve_filters


def make_retriever(mode: str, chunks: list[dict[str, Any]], embedder: GeminiEmbeddingClient, llm: GeminiClient):
    if mode == "tfidf":
        return TfidfRetriever(chunks)
    if mode == "bm25":
        return BM25Retriever(chunks)
    if mode == "dense":
        return PgVectorRetriever(embedder=embedder)
    if mode == "hybrid_rrf":
        return HybridRRFRetriever(PgVectorRetriever(embedder=embedder), BM25Retriever(chunks), candidate_k=20, rrf_k=60)
    if mode == "hybrid_rerank":
        base = HybridRRFRetriever(PgVectorRetriever(embedder=embedder), BM25Retriever(chunks), candidate_k=20, rrf_k=60)
        return GeminiRerankRetriever(base, llm=llm, candidate_k=10)
    raise ValueError(f"Unknown retriever mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agentic RAG over cases.json for a single split.")
    parser.add_argument("--chunks", type=Path, default=STARTER_DIR / "data/converted/chunks.json")
    parser.add_argument("--cases", type=Path, default=STARTER_DIR / "data/cases.json")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--retriever", choices=("tfidf", "bm25", "dense", "hybrid_rrf", "hybrid_rerank"), required=True)
    parser.add_argument("--use-filter", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--context-mode", choices=("fifo", "latest_priority"), default="fifo")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    cases = [c for c in json.loads(args.cases.read_text(encoding="utf-8")) if c.get("split") == args.split]
    if not cases:
        raise ValueError(f"No cases found for split={args.split}")

    embedder = GeminiEmbeddingClient()
    llm = GeminiClient(model_name=args.model)
    retriever = make_retriever(args.retriever, chunks, embedder, llm)
    controller = GeminiController(llm=llm, max_searches=args.max_searches)

    traces: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        controller.reset_turn_logs()
        filters = resolve_filters(case, args.use_filter)
        calls_0, pt_0, ot_0 = llm.api_calls, llm.prompt_tokens, llm.output_tokens
        emb_0 = embedder.api_calls
        t0 = time.perf_counter()
        try:
            res = run_agent(
                case["question"],
                controller,
                retriever,
                filters=filters,
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
                context_mode=args.context_mode,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            search_events = [e for e in res["events"] if e.get("event") == "search"]
            last_ctx = search_events[-1]["context"] if search_events else ""
            trace = {
                "query_id": case["id"],
                "split": args.split,
                "question": case["question"],
                "answerable": case["answerable"],
                "filter_applied": filters,
                "status": res["status"],
                "answer": res["answer"],
                "citations": res["citations"],
                "events": res["events"],
                "num_searches": len(search_events),
                "context_chars": len(last_ctx),
                "elapsed_ms": round(elapsed_ms, 2),
                "llm_calls": llm.api_calls - calls_0,
                "embedding_api_calls": embedder.api_calls - emb_0,
                "prompt_tokens": llm.prompt_tokens - pt_0,
                "output_tokens": llm.output_tokens - ot_0,
                "turn_logs": controller.turn_logs,
                "service_cost_usd": 0.0,
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            trace = {
                "query_id": case["id"],
                "split": args.split,
                "question": case["question"],
                "answerable": case["answerable"],
                "filter_applied": filters,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "answer": "",
                "citations": [],
                "events": [],
                "num_searches": 0,
                "context_chars": 0,
                "elapsed_ms": round(elapsed_ms, 2),
                "llm_calls": llm.api_calls - calls_0,
                "embedding_api_calls": embedder.api_calls - emb_0,
                "prompt_tokens": llm.prompt_tokens - pt_0,
                "output_tokens": llm.output_tokens - ot_0,
                "turn_logs": controller.turn_logs,
                "service_cost_usd": 0.0,
            }
        traces.append(trace)
        print(
            f"[{idx}/{len(cases)}] {case['id']} ({'ans' if case['answerable'] else 'unans'}) -> "
            f"status={trace['status']} searches={trace['num_searches']} cites={trace['citations']} "
            f"({trace['elapsed_ms']}ms)",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    meta = {
        "split": args.split,
        "retriever": args.retriever,
        "use_filter": args.use_filter,
        "top_k": args.top_k,
        "budget": args.budget,
        "max_searches": args.max_searches,
        "context_mode": args.context_mode,
        "llm_model": llm.model_name,
        "total_llm_calls": llm.api_calls,
        "total_prompt_tokens": llm.prompt_tokens,
        "total_output_tokens": llm.output_tokens,
        "total_embedding_api_calls": embedder.api_calls,
        "service_cost_usd": 0.0,
    }
    (args.output / "run_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved agent traces to {args.output / 'traces.json'}")


if __name__ == "__main__":
    main()
