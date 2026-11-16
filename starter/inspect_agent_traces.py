"""Print human-auditable side-by-side summary of agent answers, cited chunk texts, and gold answers."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect agent traces alongside cited chunks and expected answers.")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()

    traces = json.loads(args.traces.read_text(encoding="utf-8"))
    q_by_id = {q["id"]: q for q in json.loads(args.questions.read_text(encoding="utf-8"))}
    c_by_id = {c["id"]: c for c in json.loads(args.cases.read_text(encoding="utf-8"))}

    for t in traces:
        qid = t["query_id"]
        q_info = q_by_id.get(qid, {})
        c_info = c_by_id.get(qid, {})
        print("=" * 80)
        print(f"ID: {qid} | split={t['split']} | answerable={t['answerable']} | status={t['status']} | searches={t['num_searches']}")
        print(f"Question: {t['question']}")
        if t["answerable"]:
            print(f"Expected: {q_info.get('expected_answer')}")
            print(f"Gold loc: {q_info.get('gold')}")
        else:
            print(f"Absence reason: {c_info.get('absence_reason')}")
        print(f"Agent Answer: {t['answer']}")
        print(f"Citations: {t['citations']}")

        # Map available context chunks across search events
        ctx_chunks = {}
        for ev in t.get("events", []):
            if ev.get("event") == "search":
                print(f"  [search] query={ev.get('query')!r} filter={ev.get('filter')} ctx_ids={ev.get('context_ids')}")
                for r in ev.get("results", []):
                    ctx_chunks[r["id"]] = r["text"]
        for cid in t["citations"]:
            txt = ctx_chunks.get(cid, "<NOT IN SEARCH RESULTS>")
            print(f"  [cited chunk {cid}]: {txt.replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
