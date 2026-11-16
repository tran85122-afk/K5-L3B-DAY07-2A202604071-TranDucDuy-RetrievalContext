"""Export Day 6 strategy-B chunks from PostgreSQL and prepare Day 7 datasets (24 cases)."""
import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import psycopg
from dotenv import load_dotenv

STARTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = STARTER_DIR.parent
sys.path.insert(0, str(STARTER_DIR))

from baseline import validate
from convert_day6 import convert_chunk, convert_question


DEFAULT_DAY6_QUESTIONS = (
    REPO_ROOT.parent
    / "K5-L3B-DAY06-TranDucDuy-2A202604071-DataIngestion"
    / "benchmark"
    / "questions.json"
)

# 5 brand-new independent test questions from untouched pages of wdr2021.pdf
# (avoiding all dev pages and Day 6's previously inspected test pages).
NEW_TEST_QUESTIONS = [
    {
        "id": "t16",
        "split": "test",
        "type": "direct",
        "question": "By what date and by how much did GPS location analysis show social distancing policies reduced mobility in major US cities?",
        "expected_answer": "By March 23, 2020, social distancing policies had helped reduce mobility in major US cities by half.",
        "gold_source_id": "wdr2021",
        "gold_page": 56,
        "evidence_quote": "Analysis of GPS locations showed that by March 23, 2020, social distancing policies had helped reduce mobility in major US cities by half.",
        "filter": {"chapter": "1"},
    },
    {
        "id": "t17",
        "split": "test",
        "type": "direct",
        "question": "In 2013, how many people globally were estimated to be missing from population census frames?",
        "expected_answer": "Between 170 million and 320 million people, with the poorest more likely to be missed.",
        "gold_source_id": "wdr2021",
        "gold_page": 80,
        "evidence_quote": "in 2013 it was estimated that globally between 170 million and 320 million people were missing from population census frames, with the poorest more likely to be missed.",
    },
    {
        "id": "t18",
        "split": "test",
        "type": "paraphrase",
        "question": "Which individual factor had the largest positive effect on the likelihood of internet uptake in the 22 developing countries surveyed in 2017/18?",
        "expected_answer": "Education: completion of primary education adds 35 percentage points to the likelihood of internet uptake.",
        "gold_source_id": "wdr2021",
        "gold_page": 184,
        "evidence_quote": "Of these, the largest effect is associated with education: completion of primary education adds 35 percentage points to the likelihood of internet uptake.",
    },
    {
        "id": "t19",
        "split": "test",
        "type": "direct",
        "question": "Which two African countries updated their data protection legal regimes in 2019 using instruments reflecting elements of the GDPR?",
        "expected_answer": "Kenya (new Data Protection Act entered into force in 2019) and Nigeria (Data Protection Regulation issued by NITDA).",
        "gold_source_id": "wdr2021",
        "gold_page": 215,
        "evidence_quote": "In 2019 Kenya’s new Data Protection Act entered into force, while Nigeria’s National Information Technology Development Agency issued a new Data Protection Regulation. Both instruments reflect many of the elements contained in the GDPR.",
        "filter": {"chapter": "6"},
    },
    {
        "id": "t20",
        "split": "test",
        "type": "condition",
        "question": "Which two countries did EITI delist in 2010, and for what reason?",
        "expected_answer": "Equatorial Guinea and São Tomé and Príncipe, for insufficient progress against agreed-on deadlines.",
        "gold_source_id": "wdr2021",
        "gold_page": 295,
        "evidence_quote": "EITI delisted Equatorial Guinea and São Tomé and Príncipe in 2010 for insufficient progress against agreed-on deadlines.",
    },
]

UNANSWERABLE_CASES = [
    {
        "id": "q16",
        "split": "dev",
        "question": "What are the maximum administrative fine tiers (in euros and percentage of global annual turnover) under the EU Artificial Intelligence Act (EU AI Act)?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Verified across all 1,495 chunks (201 extracted pages of WDR 2021): 'AI Act' / 'Artificial Intelligence Act' has 0 occurrences; WDR 2021 was published in 2021 before the EU AI Act.",
    },
    {
        "id": "q17",
        "split": "dev",
        "question": "Under Vietnam's Decree 13/2023/ND-CP on Personal Data Protection, within how many hours must a data controller notify the authority of a data breach?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Verified across all 1,495 chunks: 'Decree 13' has 0 occurrences; the 5 chunks mentioning Vietnam (pages 35, 260, 261, 268, 269) discuss trade/e-commerce/statistics, not Decree 13/2023/ND-CP.",
    },
    {
        "id": "t21",
        "split": "test",
        "question": "Does the Report mandate ISO/IEC 27001 or SOC 2 Type II certification for national statistical offices?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Verified across all 1,495 chunks: '27001' and 'SOC 2' have 0 occurrences in the entire corpus.",
    },
    {
        "id": "t22",
        "split": "test",
        "question": "What average download speed (in Mbps) and monthly subscription price did SpaceX Starlink achieve in Sub-Saharan Africa according to Chapter 5?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Verified across all 1,495 chunks: 'Starlink' and 'SpaceX' have 0 occurrences; chunk wdr2021:p183:p5 mentions low-earth orbit satellites conceptually without Starlink speeds or prices.",
    },
]


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to single space while mapping each char to source index."""
    norm_chars: list[str] = []
    index_map: list[int] = []
    in_space = True
    nfc = unicodedata.normalize("NFC", text)
    for idx, ch in enumerate(nfc):
        if ch.isspace():
            if not in_space:
                norm_chars.append(" ")
                index_map.append(idx)
                in_space = True
        else:
            norm_chars.append(ch.casefold())
            index_map.append(idx)
            in_space = False
    if norm_chars and norm_chars[-1] == " ":
        norm_chars.pop()
        index_map.pop()
    return "".join(norm_chars), index_map


def extract_verbatim_evidence(chunk_text: str, quote: str) -> str | None:
    """Return exact substring in chunk_text matching quote after whitespace normalization."""
    if quote in chunk_text:
        return quote
    norm_chunk, idx_map = _normalize_with_map(chunk_text)
    norm_quote = re.sub(r"\s+", " ", unicodedata.normalize("NFC", quote)).strip().casefold()
    pos = norm_chunk.find(norm_quote)
    if pos == -1:
        return None
    start = idx_map[pos]
    end_pos = pos + len(norm_quote) - 1
    end = idx_map[end_pos] + 1
    nfc_chunk = unicodedata.normalize("NFC", chunk_text)
    return nfc_chunk[start:end]


def export_day6_chunks(database_url: str, strategy: str = "B") -> list[dict]:
    """Query all chunks for the chosen Day 6 strategy from PostgreSQL."""
    with psycopg.connect(database_url, connect_timeout=5) as conn:
        rows = conn.execute(
            """
            SELECT chunk_id, document_id, source, page, text, metadata
            FROM chunks
            WHERE strategy = %s
            ORDER BY page, chunk_id
            """,
            (strategy,),
        ).fetchall()
    return [
        {
            "chunk_id": r[0],
            "document_id": r[1],
            "source": r[2],
            "page": r[3],
            "text": r[4],
            "metadata": r[5],
        }
        for r in rows
    ]


def verify_absences(chunks: list[dict]) -> None:
    """Confirm unanswerable topics are absent from all exported chunks."""
    checks = {
        "ai act": [],
        "decree 13": [],
        "27001": [],
        "soc 2": [],
        "starlink": [],
        "spacex": [],
    }
    for c in chunks:
        low = c["text"].lower()
        for kw in checks:
            if kw in low:
                checks[kw].append(c["chunk_id"])
    non_empty = {k: v for k, v in checks.items() if v}
    if non_empty:
        raise RuntimeError(f"Unanswerable keyword unexpectedly found: {non_empty}")
    for cid in ("wdr2021:p183:p5", "wdr2021:p35:p0", "wdr2021:p260:p8", "wdr2021:p261:p1", "wdr2021:p268:p5", "wdr2021:p269:p1"):
        chunk = next(c for c in chunks if c["chunk_id"] == cid)
        print(f"[near-miss check] {cid}: {chunk['text'][:220].replace(chr(10), ' ')}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Day 7 chunks, questions, and 24 agent cases from Day 6.")
    parser.add_argument("--day6-questions", type=Path, default=DEFAULT_DAY6_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=STARTER_DIR / "data")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing DATABASE_URL in .env")

    raw_chunks = export_day6_chunks(db_url, strategy="B")
    if not raw_chunks:
        raise RuntimeError("No strategy='B' chunks found in PostgreSQL")
    print(f"Exported {len(raw_chunks)} chunks (strategy B) from PostgreSQL.")

    verify_absences(raw_chunks)

    day6_questions = json.loads(args.day6_questions.read_text(encoding="utf-8"))
    dev_questions = [q for q in day6_questions if q["split"] == "dev"]
    if len(dev_questions) != 15:
        raise RuntimeError(f"Expected 15 dev questions from Day 6, got {len(dev_questions)}")

    raw_questions = []
    for q in dev_questions + NEW_TEST_QUESTIONS:
        q_copy = dict(q)
        page_chunks = [
            c for c in raw_chunks
            if c["document_id"] == q_copy["gold_source_id"] and c["page"] == q_copy["gold_page"]
        ]
        matched = None
        for c in page_chunks:
            verbatim = extract_verbatim_evidence(c["text"], q_copy["evidence_quote"])
            if verbatim is not None:
                matched = verbatim
                break
        if matched is None:
            raise RuntimeError(f"Question {q_copy['id']}: could not match evidence on page {q_copy['gold_page']}")
        q_copy["evidence_quote"] = matched
        raw_questions.append(q_copy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_chunks_path = args.output_dir / "day6_chunks_raw.json"
    raw_questions_path = args.output_dir / "day6_questions_raw.json"
    raw_chunks_path.write_text(json.dumps(raw_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_questions_path.write_text(json.dumps(raw_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    converted_dir = args.output_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    chunks = [convert_chunk(c) for c in raw_chunks]
    questions = [convert_question(q) for q in raw_questions]
    validate(chunks, questions)

    (converted_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (converted_dir / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Build 24-case dataset for agent evaluation (no gold or expected_answer leaked to agent).
    cases = []
    for q in questions:
        case = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
        }
        if "filter" in q:
            case["filter"] = q["filter"]
        cases.append(case)
    cases.extend(UNANSWERABLE_CASES)
    cases.sort(key=lambda c: (0 if c["split"] == "dev" else 1, c["id"]))
    (args.output_dir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dev_n = sum(1 for c in cases if c["split"] == "dev")
    test_n = sum(1 for c in cases if c["split"] == "test")
    print(f"Prepared converted/chunks.json ({len(chunks)}), converted/questions.json ({len(questions)}), cases.json ({dev_n} dev, {test_n} test).")


if __name__ == "__main__":
    main()
