"""Unit tests for Day 7 retrieval extensions (BM25, Hybrid RRF, latest_priority context, verbatim matcher)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from agent_adapters import GeminiController
from agentic import run_agent
from build_advanced_index import classify_element, compact_layout, extract_diagram_xy_columns
from evaluate_advanced import evaluate_with_reviews
from prepare_day7_data import extract_verbatim_evidence
from retrievers import BM25Retriever, HybridRRFRetriever
from run_advanced_benchmark import AdvancedMultiModalRetriever


def make_chunk(cid: str, text: str, page: int = 1, chapter: str = "1") -> dict:
    return {
        "id": cid,
        "source_id": "wdr2021",
        "source": "starter/data/wdr2021.pdf",
        "page": page,
        "text": text,
        "metadata": {"chapter": chapter},
    }


class Day7ExtensionTests(unittest.TestCase):
    def test_bm25_and_metadata_filter(self):
        chunks = [
            make_chunk("c1", "Data governance framework in chapter two.", page=86, chapter="2"),
            make_chunk("c2", "Data governance framework and antitrust cases.", page=250, chapter="7"),
        ]
        bm25 = BM25Retriever(chunks)
        hits = bm25.search("antitrust data governance", filters={"chapter": "7"}, top_k=3)
        self.assertEqual([h["id"] for h in hits], ["c2"])

    def test_hybrid_rrf_fuses_dense_and_lexical(self):
        chunks = [
            make_chunk("c1", "Unrelated text with keyword overlap data.", page=10),
            make_chunk("c2", "Private intent data improves timeliness and granularity.", page=48),
            make_chunk("c3", "Another passage about public statistics.", page=80),
        ]
        bm25 = BM25Retriever(chunks)

        class StubDense:
            def search(self, query, *, filters=None, top_k=3):
                return [
                    {**chunks[1], "score": 0.91},
                    {**chunks[2], "score": 0.85},
                ][:top_k]

        hybrid = HybridRRFRetriever(StubDense(), bm25, candidate_k=3, rrf_k=60)
        hits = hybrid.search("timeliness granularity private data", top_k=2)
        self.assertEqual(hits[0]["id"], "c2")
        self.assertEqual(hits[0]["strategy"], "hybrid_rrf")

    def test_latest_priority_context_prevents_starvation(self):
        stale = make_chunk("stale", "X" * 120, page=1)
        fresh = make_chunk("fresh", "Exact answer 42.", page=2)

        class TwoStepController:
            def __init__(self):
                self.turn = 0

            def decide(self, question, history):
                self.turn += 1
                if self.turn == 1:
                    return {"action": "search", "query": "first"}
                if self.turn == 2:
                    return {"action": "search", "query": "second"}
                # On turn 3, fresh must be present inside context_ids despite tight budget.
                assert "fresh" in history[-1]["context_ids"]
                return {"action": "answer", "answer": "42 [fresh]", "citations": ["fresh"]}

        class TwoStepRetriever:
            def __init__(self):
                self.calls = 0

            def search(self, query, *, filters=None, top_k=3):
                self.calls += 1
                return [stale] if self.calls == 1 else [fresh]

        # Budget 180 fits one chunk (~165 chars with header) but not both.
        # Under fifo, stale fills the budget and fresh cannot enter context.
        with self.assertRaises(AssertionError):
            run_agent(
                "What is the answer?",
                TwoStepController(),
                TwoStepRetriever(),
                budget=180,
                context_mode="fifo",
            )
        # Under latest_priority, fresh from the second search replaces stale at the front.
        res = run_agent(
            "What is the answer?",
            TwoStepController(),
            TwoStepRetriever(),
            budget=180,
            context_mode="latest_priority",
        )
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["citations"], ["fresh"])

    def test_extract_verbatim_evidence_across_newlines(self):
        chunk_text = "Prefix words.\nThe phrase “sharing and reuse” is shorthand used\nin this Report for data."
        quote = "The phrase “sharing and reuse” is shorthand used in this Report"
        matched = extract_verbatim_evidence(chunk_text, quote)
        self.assertIsNotNone(matched)
        self.assertIn(matched, chunk_text)
        self.assertIn("\n", matched)

    def test_advanced_index_compaction_and_classification(self):
        raw = "  Model          Policy goal        Cost  \n\n  Open       Business freedom    Lower  \n"
        self.assertEqual(compact_layout(raw), "Model | Policy goal | Cost\nOpen | Business freedom | Lower")
        self.assertEqual(classify_element("Table 7.1", ""), "table")
        self.assertEqual(classify_element("Figure 5.6", ""), "chart")
        self.assertEqual(classify_element("Figure 1.2", ""), "diagram")
        self.assertEqual(classify_element("Map 7.1", ""), "diagram")

        class StubPage:
            def extract_text(self, visitor_text=None):
                samples = [
                    (50.0, 500.0, "Figure 9.3 Steps to integrate"),
                    (120.0, 450.0, "Stage A"),
                    (320.0, 450.0, "Stage B"),
                    (120.0, 430.0, "Box A1"),
                    (320.0, 430.0, "Box B1"),
                    (50.0, 380.0, "Source: WDR 2021 team."),
                ]
                for x, y, txt in samples:
                    visitor_text(txt, None, [1, 0, 0, 1, x, y], None, 10)
                return ""

        grouped = extract_diagram_xy_columns(StubPage(), "Figure 9.3")
        self.assertIn("[Diagram Column 1", grouped)
        self.assertIn("Stage A", grouped)
        self.assertLess(grouped.index("Box A1"), grouped.index("Stage B"))

    def test_advanced_multimodal_retriever_element_boost_and_filter(self):
        vis_table = {
            "id": "wdr2021:p83:vis_table_2_1",
            "source_id": "wdr2021",
            "source": "starter/data/wdr2021.pdf",
            "page": 83,
            "text": "[TABLE] Table 2.1: Openness\nOpenness score | 38 | 66",
            "metadata": {"chapter": "chapter_2", "element_type": "table", "element_id": "Table 2.1"},
        }
        vis_chart = {
            "id": "wdr2021:p296:vis_figure_8_4",
            "source_id": "wdr2021",
            "source": "starter/data/wdr2021.pdf",
            "page": 296,
            "text": "[CHART] Figure 8.4: Data protection\nFigure 8.5: Operational antitrust=75%",
            "metadata": {"chapter": "chapter_8", "element_type": "chart", "element_id": "Figure 8.4"},
        }
        prose = make_chunk("wdr2021:p10:p0", "General prose mention of statistical capacity.", page=10, chapter="1")
        retriever = AdvancedMultiModalRetriever([prose, vis_table, vis_chart], dense_retriever=None)
        hits = retriever.search("Compare Table 2.1 and Figure 8.5 indicators", top_k=2)
        self.assertEqual({h["id"] for h in hits}, {"wdr2021:p83:vis_table_2_1", "wdr2021:p296:vis_figure_8_4"})
        filtered = retriever.search("Compare Table 2.1 and Figure 8.5", filters={"chapter": "chapter_2"}, top_k=2)
        self.assertEqual([h["id"] for h in filtered], ["wdr2021:p83:vis_table_2_1"])
        # Explicit PDF page boost prioritizes page-matched chunk over adjacent-page distractor
        adjacent = make_chunk("wdr2021:p114:p0", "Agricultural IoT sensors measure moisture.", page=114, chapter="3")
        exact_page = make_chunk("wdr2021:p115:p0", "Agricultural commodities IoT commodity weight.", page=115, chapter="3")
        p_ret = AdvancedMultiModalRetriever([adjacent, exact_page], dense_retriever=None)
        p_hits = p_ret.search("On page 115, agricultural IoT sensors", top_k=1)
        self.assertEqual(p_hits[0]["id"], "wdr2021:p115:p0")

    def test_evaluate_advanced_enforces_strict_review_consistency(self):
        bm = [
            {"id": "a1", "split": "test", "primary_group": "table", "answerable": True},
            {"id": "u1", "split": "test", "primary_group": "unanswerable", "answerable": False},
        ]
        traces = [
            {"query_id": "a1", "status": "answered", "events": [{"event": "search"}], "citations": ["c1"], "elapsed_ms": 100.0},
            {"query_id": "u1", "status": "abstain", "events": [{"event": "search"}], "citations": [], "elapsed_ms": 200.0},
        ]
        good_reviews = [
            {"query_id": "a1", "pass": True, "answer_grade": 2, "citation_support": [True]},
            {"query_id": "u1", "pass": True, "citation_support": [], "abstention_correct": True},
        ]
        summary = evaluate_with_reviews(bm, traces, good_reviews, "test")
        self.assertEqual(summary["passed"], 2)
        self.assertTrue(summary["meets_90pct_target"])
        bad_reviews = [
            good_reviews[0],
            {"query_id": "u1", "pass": True, "citation_support": [], "abstention_correct": False},
        ]
        with self.assertRaises(ValueError):
            evaluate_with_reviews(bm, traces, bad_reviews, "test")

    def test_gemini_controller_unwraps_single_item_list_json(self):
        class StubLLM:
            prompt_tokens = 10
            output_tokens = 5

            def generate_json(self, system_prompt, user_prompt, temperature=0.0):
                self.prompt_tokens += 10
                self.output_tokens += 5
                return [{"action": "search", "query": "Table 8.3 intermediaries"}]

        ctrl = GeminiController(llm=StubLLM(), max_searches=4)
        action = ctrl.decide("Test question", [])
        self.assertEqual(action, {"action": "search", "query": "Table 8.3 intermediaries"})


if __name__ == "__main__":
    unittest.main()
