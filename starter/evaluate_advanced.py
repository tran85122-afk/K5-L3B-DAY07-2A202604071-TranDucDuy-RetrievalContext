"""Inspect and aggregate Day 7 Advanced Multimodal benchmark traces with manual audit reviews."""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

STARTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STARTER_DIR))

from agent_metrics import aggregate as base_aggregate


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return round(sorted_vals[lo], 2)
    return round(sorted_vals[lo] * (hi - idx) + sorted_vals[hi] * (idx - lo), 2)


def dump_for_audit(benchmark_cases: list[dict[str, Any]], traces: list[dict[str, Any]], split: str) -> str:
    bm_split = [c for c in benchmark_cases if c.get("split") == split]
    trace_by_id = {t["query_id"]: t for t in traces}
    lines: list[str] = []
    for bm in bm_split:
        qid = bm["id"]
        t = trace_by_id.get(qid, {})
        lines.append(f"=== {qid} [{bm['primary_group']} | answerable={bm['answerable']}] ===")
        lines.append(f"Q: {bm['question']}")
        lines.append(f"GOLD_CITES: {bm.get('gold_chunk_ids', [])} | PAGES: {bm.get('gold_pages', [])}")
        lines.append(f"REQUIRED_FACTS: {bm.get('required_facts', [])}")
        lines.append(f"EXPECTED: {bm.get('expected_answer', '')}")
        lines.append(
            f"AGENT_STATUS: {t.get('status')} | SEARCHES: {t.get('num_searches')} | "
            f"CITES: {t.get('citations', [])} | MS: {t.get('elapsed_ms')}"
        )
        lines.append(f"AGENT_ANSWER: {t.get('answer', '')}")
        lines.append("")
    return "\n".join(lines)


def evaluate_with_reviews(
    benchmark_cases: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    bm_split = [c for c in benchmark_cases if c.get("split") == split]
    if not bm_split:
        raise ValueError(f"No benchmark cases found for split={split}")

    base = base_aggregate(bm_split, traces, reviews)
    rev_by_id = {r["query_id"]: r for r in reviews}
    trace_by_id = {t["query_id"]: t for t in traces}

    by_group: dict[str, dict[str, Any]] = defaultdict(lambda: {"passed": 0, "total": 0})
    passed_total = 0
    latencies: list[float] = []
    llm_calls = 0
    emb_calls = 0
    prompt_tokens = 0
    output_tokens = 0
    total_searches = 0
    total_cost = 0.0

    for bm in bm_split:
        qid = bm["id"]
        grp = bm["primary_group"]
        t = trace_by_id[qid]
        r = rev_by_id[qid]
        ok = bool(r.get("pass"))
        # Cross-check strict consistency between pass bool and base review fields
        if t.get("status") == "error":
            ok = False
        elif bm["answerable"]:
            if ok and (r.get("answer_grade") != 2 or not r.get("citation_support") or not all(r["citation_support"])):
                raise ValueError(f"{qid}: pass=True requires answer_grade==2 and all citation_support True")
        else:
            if ok and not r.get("abstention_correct"):
                raise ValueError(f"{qid}: pass=True requires abstention_correct==True")

        by_group[grp]["total"] += 1
        by_group[grp]["passed"] += int(ok)
        passed_total += int(ok)

        latencies.append(float(t.get("elapsed_ms", 0.0)))
        llm_calls += int(t.get("llm_calls", 0))
        emb_calls += int(t.get("embedding_api_calls", 0))
        prompt_tokens += int(t.get("prompt_tokens", 0))
        output_tokens += int(t.get("output_tokens", 0))
        total_searches += int(t.get("num_searches", 0))
        total_cost += float(t.get("service_cost_usd", 0.0))

    lat_sorted = sorted(latencies)
    for grp, gdata in by_group.items():
        gdata["pass_rate"] = round(gdata["passed"] / gdata["total"], 4) if gdata["total"] else 0.0

    return {
        "split": split,
        "total_cases": len(bm_split),
        "passed": passed_total,
        "pass_rate": round(passed_total / len(bm_split), 4),
        "meets_90pct_target": (passed_total / len(bm_split)) >= 0.90,
        "by_group": dict(by_group),
        "base_agent_metrics": base,
        "telemetry": {
            "total_searches": total_searches,
            "avg_searches": round(total_searches / len(bm_split), 2),
            "llm_calls": llm_calls,
            "embedding_api_calls": emb_calls,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "service_cost_usd": round(total_cost, 6),
            "latency_ms_p50": _percentile(lat_sorted, 0.50),
            "latency_ms_p95": _percentile(lat_sorted, 0.95),
            "latency_ms_max": round(max(lat_sorted), 2) if lat_sorted else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump or evaluate Advanced Multimodal traces.")
    parser.add_argument("--benchmark", type=Path, default=STARTER_DIR / "data/advanced/benchmark_advanced.json")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test", "test_v2"), required=True)
    parser.add_argument("--dump", type=Path, help="Write readable side-by-side audit comparison file.")
    parser.add_argument("--reviews", type=Path, help="Verified manual reviews.json path.")
    parser.add_argument("--output", type=Path, help="Output summary.json path.")
    args = parser.parse_args()

    benchmark_cases = json.loads(args.benchmark.read_text(encoding="utf-8"))
    traces = json.loads(args.traces.read_text(encoding="utf-8"))

    if args.dump:
        text = dump_for_audit(benchmark_cases, traces, args.split)
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(text, encoding="utf-8")
        print(f"Wrote audit dump to {args.dump}")

    if args.reviews:
        reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
        summary = evaluate_with_reviews(benchmark_cases, traces, reviews, args.split)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
