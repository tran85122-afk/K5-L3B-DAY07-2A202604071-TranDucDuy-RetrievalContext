# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo cá nhân](report/individual/2A202604071_TranDucDuy.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.

---

## Cấu trúc bài làm & Hướng dẫn chạy tái lập (Trần Đức Duy — 2A202604071)

### Thành phần triển khai
- [starter/prepare_day7_data.py](starter/prepare_day7_data.py): Kết nối PostgreSQL + `pgvector` của Day 6 (`strategy='B'`, 1.495 chunks, `gemini-embedding-001` 768 dims), xuất chunks, chuẩn hóa khớp nguyên văn `evidence_quote`, sinh bộ câu hỏi `20` câu có đáp án và `starter/data/cases.json` (`24` câu: `17` dev + `7` test độc lập mới).
- [starter/retrievers.py](starter/retrievers.py): Cài đặt `TfidfRetriever`, `BM25Retriever`, `PgVectorRetriever` (Dense Day 6), `HybridRRFRetriever` (Dense + BM25 dùng Reciprocal Rank Fusion `k=60`), `GeminiRerankRetriever` (listwise LLM reranker) và `GeminiEmbeddingClient` / `GeminiClient` có rate-limit throttle + retry + embedding cache.
- [starter/run_retrieval_experiments.py](starter/run_retrieval_experiments.py): Chạy thực nghiệm và đo `Hit@1`, `Evidence Hit@3`, `Context Evidence Rate`, `MRR@3`, thời gian và token/chi phí trên từng split (`dev` hoặc `test`).
- [starter/agentic.py](starter/agentic.py) & [starter/agent_adapters.py](starter/agent_adapters.py): Vòng lặp Agentic RAG `run_agent` (bổ sung tùy chọn `context_mode='latest_priority'` chống nghẽn context khi tìm nhiều lượt) và `GeminiController` điều phối tool `search` / `answer` / `abstain`.
- [starter/run_agent_benchmark.py](starter/run_agent_benchmark.py) & [starter/inspect_agent_traces.py](starter/inspect_agent_traces.py): Chạy Agent trên `cases.json` và hiển thị song song câu trả lời, chunk trích dẫn và đáp án chuẩn để chấm thủ công `reviews.json`.
- [starter/build_advanced_index.py](starter/build_advanced_index.py), [starter/prepare_advanced_benchmark.py](starter/prepare_advanced_benchmark.py), [starter/run_advanced_benchmark.py](starter/run_advanced_benchmark.py) & [starter/evaluate_advanced.py](starter/evaluate_advanced.py): Đường ống thử thách nâng cao (+15đ) trích xuất `1,589` chunks (`1,495` text + `94` bảng/biểu đồ/sơ đồ tái cấu trúc cột 2D XY), bộ `78` câu `benchmark_advanced.json` (`18` dev + `30` test v1 đã dùng để chỉnh + `30` test_v2 độc lập; kết quả chính thức test_v2 = `24/30`, chưa đạt ngưỡng 90%), truy xuất `AdvancedMultiModalRetriever` và kiểm toán nhị phân nghiêm ngặt `reviews.json` -> `summary.json`.
- [tests/test_day7_extensions.py](tests/test_day7_extensions.py): Unit tests bổ sung cho BM25, Hybrid RRF, `latest_priority` context mode, chuẩn hóa evidence, phân loại/tái cấu trúc XY sơ đồ và đánh giá nâng cao.

### Lệnh chạy nhanh

```powershell
# 1. Cài đặt môi trường & cấu hình .env (copy từ .env.example, điền GEMINI_API_KEY và DATABASE_URL)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Chạy toàn bộ unit tests (23 tests)
python -m unittest discover -s tests -v

# 3. Xuất chunks & tạo benchmark 24 câu từ PostgreSQL Day 6
python starter/prepare_day7_data.py

# 4. Chạy benchmark các phương pháp Retrieval trên dev (rồi mới chạy trên test sau khi chốt)
python starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
python starter/run_retrieval_experiments.py --split dev
python starter/run_retrieval_experiments.py --split test --configs tfidf bm25 dense hybrid_rrf hybrid_rrf_filter_budget1800

# 5. Chạy Agentic RAG (Baseline vs Selected) và tổng hợp nhãn chấm thủ công
python starter/run_agent_benchmark.py --split dev --retriever tfidf --budget 800 --context-mode fifo --output starter/results/agent-dev-baseline
python starter/run_agent_benchmark.py --split dev --retriever hybrid_rrf --use-filter --budget 1800 --context-mode latest_priority --output starter/results/agent-dev-selected
python starter/run_agent_benchmark.py --split test --retriever tfidf --budget 800 --context-mode fifo --output starter/results/agent-test-baseline
python starter/run_agent_benchmark.py --split test --retriever hybrid_rrf --use-filter --budget 1800 --context-mode latest_priority --output starter/results/agent-test-selected
python starter/agent_metrics.py --cases starter/data/cases.json --traces starter/results/agent-test-selected/traces.json --reviews starter/results/agent-test-selected/reviews.json --split test --output starter/results/agent-test-selected/summary.json

# 6. Chạy thử thách nâng cao (+15 điểm bonus): Index đa phương thức & Benchmark 48 câu
python starter/build_advanced_index.py
python starter/prepare_advanced_benchmark.py
python starter/run_advanced_benchmark.py --split dev --output starter/results/advanced-dev
python starter/run_advanced_benchmark.py --split test --output starter/results/advanced-test
python starter/evaluate_advanced.py --traces starter/results/advanced-test/traces.json --reviews starter/results/advanced-test/reviews.json --split test --output starter/results/advanced-test/summary.json
# test v1 ở trên đã bị dùng để chỉnh pipeline -> đánh giá độc lập chính thức là test_v2 (30 câu mới, chạy 1 lần):
python starter/run_advanced_benchmark.py --split test_v2 --output starter/results/advanced-test-v2
python starter/evaluate_advanced.py --traces starter/results/advanced-test-v2/traces.json --reviews starter/results/advanced-test-v2/reviews.json --split test_v2 --output starter/results/advanced-test-v2/summary.json
```

