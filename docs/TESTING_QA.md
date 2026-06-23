# Testing & QA Matrix — Local RAG LLM

> **Purpose:** Verify the correctness, accuracy, safety, and performance of the local RAG pipeline before trusting it for real document workflows.

---

## How to Run Tests

All tests below are **manual black-box tests** executed through the Streamlit UI (`http://localhost:8501`). No test framework is required.

**Setup for each test session:**
1. Start Ollama: `ollama serve`
2. Launch the app: `streamlit run app.py`
3. Upload the designated test PDF(s) via the sidebar
4. Click **Process**, wait for confirmation, then run the queries below

---

## Test Matrix

### TC-01 — Out-of-Bounds Query (Scope Rejection)

| Field | Detail |
|---|---|
| **Objective** | Confirm the model refuses to answer questions not covered by the uploaded documents and does not hallucinate an answer from its training data. |
| **Category** | Boundary / Safety |
| **Test PDF** | Any single-topic PDF (e.g., a product manual for a coffee maker) |
| **Query** | `"What is the capital of France?"` |
| **Expected Response** | A refusal such as: *"I could not find an answer to that in the uploaded documents."* The model must NOT answer "Paris" from its training knowledge. |
| **Failure Condition** | The model answers using external knowledge instead of restricting itself to the document context. |
| **Pass Criteria** | Response explicitly states the information is not in the uploaded documents. Source citations panel shows 0 or irrelevant sources. |
| **Notes** | This test validates the `SYSTEM_PROMPT` hallucination guard. If it fails, tighten the system prompt or reduce the model's `temperature` parameter toward `0.0`. |

---

### TC-02 — Factual Accuracy / Hallucination Check

| Field | Detail |
|---|---|
| **Objective** | Verify the model returns factually correct answers that are directly traceable to a specific passage in the uploaded document. |
| **Category** | Accuracy / Hallucination Detection |
| **Test PDF** | A PDF with a clearly stated, verifiable fact (e.g., a company policy doc stating: *"Annual leave entitlement is 20 days per year for all full-time employees."*) |
| **Query** | `"How many annual leave days are full-time employees entitled to?"` |
| **Expected Response** | `"Full-time employees are entitled to 20 days of annual leave per year."` with the correct source file and page number cited. |
| **Failure Condition** | The model returns a different number, omits the citation, or adds unrequested qualifications not present in the document. |
| **Pass Criteria** | Answer matches the document verbatim or as a direct paraphrase. Source citation expands to show the correct filename and page. |
| **Notes** | Manually open the source PDF and verify the cited page contains the relevant text. Repeat with 3 different factual questions from the same document. |

---

### TC-03 — Multi-Document Query (Cross-Document Retrieval)

| Field | Detail |
|---|---|
| **Objective** | Confirm the retriever correctly identifies and combines relevant context from two different uploaded PDFs. |
| **Category** | Retrieval Correctness |
| **Test PDFs** | **Doc A:** A network security policy PDF. **Doc B:** A remote access policy PDF. |
| **Query** | `"What are the rules for VPN access and password complexity?"` |
| **Expected Response** | A synthesized answer drawing from both documents, with source citations referencing both Doc A and Doc B. |
| **Failure Condition** | The answer references only one document, ignores relevant content from the other, or returns a hallucinated policy not present in either file. |
| **Pass Criteria** | Both filenames appear in the Sources panel. The answer content is traceable to passages in both documents. |
| **Notes** | Adjust `TOP_K_RESULTS` in `app.py` (default: 4) upward if cross-document retrieval misses relevant chunks. |

---

### TC-04 — Speed / Latency Benchmark

| Field | Detail |
|---|---|
| **Objective** | Measure and document end-to-end response latency under typical usage conditions to set realistic user expectations. |
| **Category** | Performance |
| **Test PDF** | A 10-page PDF (~4,000 words) |
| **Setup** | Time measurement starts when the user presses Enter on the query and ends when the full response is displayed. Use a stopwatch or browser DevTools Network tab. |
| **Query 1 (short answer)** | `"What is the document about?"` |
| **Query 2 (long answer)** | `"Provide a detailed summary of all the main topics covered in this document."` |
| **Expected Latency (modern laptop CPU, M1/M2/Intel i7)** | Query 1: 5–15 seconds · Query 2: 15–40 seconds |
| **Failure Condition** | Response takes longer than 120 seconds, the app freezes, or Ollama returns a timeout error. |
| **Pass Criteria** | Responses are returned within the expected range without app crash. |
| **Notes** | Log results in the table below. If latency exceeds 60s, consider switching to a smaller quantized variant (e.g., `llama3:8b-instruct-q4_K_M`). |

**Latency Log Table (fill in during testing):**

| Run | Query Type | PDF Pages | Response Time (sec) | Pass/Fail |
|---|---|---|---|---|
| 1 | Short | 10 | | |
| 2 | Long | 10 | | |
| 3 | Short | 50 | | |
| 4 | Long | 50 | | |

---

### TC-05 — Empty / Nonsense Input Handling

| Field | Detail |
|---|---|
| **Objective** | Ensure the application handles malformed, empty, or adversarial queries gracefully without crashing. |
| **Category** | Edge Case / Robustness |
| **Test PDF** | Any uploaded PDF |
| **Queries to test** | 1. Single space character ` ` · 2. `"???"` · 3. A 500-word block of random text pasted as a query · 4. Emoji only: `"🔥🔥🔥"` |
| **Expected Response** | A graceful response such as *"I could not find an answer to that in the uploaded documents"* or a relevant partial answer. The app must NOT crash, throw an unhandled exception, or display a Python traceback to the user. |
| **Failure Condition** | Streamlit displays a raw Python exception. The app becomes unresponsive. |
| **Pass Criteria** | All four queries return a user-readable response. The sidebar status indicator remains green. |
| **Notes** | Errors are caught by the `try/except` block in `app.py`. If a traceback leaks, wrap the `rag_chain.invoke()` call in a more specific exception handler. |

---

### TC-06 — Database Persistence (Restart Resilience)

| Field | Detail |
|---|---|
| **Objective** | Verify that indexed documents survive an application restart without requiring re-upload or re-embedding. |
| **Category** | Data Persistence |
| **Steps** | 1. Upload 2 PDFs and click Process. Note the chunk count shown in the sidebar. 2. Stop the Streamlit app (`Ctrl+C`). 3. Restart the app: `streamlit run app.py`. 4. Do NOT upload any files. 5. Ask a question answered by the previously uploaded PDFs. |
| **Expected Response** | The sidebar auto-detects the existing ChromaDB collection and displays *"Existing DB found (N chunks)"*. The query returns a correct answer with source citations. |
| **Failure Condition** | The app reports no documents indexed after restart, or the query returns an out-of-bounds refusal. |
| **Pass Criteria** | Chunk count matches the pre-restart count. Query response is accurate with correct source citations. `.chroma/` directory exists on disk and is non-empty. |
| **Notes** | This validates the `load_existing_vector_store()` function in `app.py`. Check that `.chroma/` is not accidentally listed in `.gitignore` if persistence is required across machines (note: by default it IS gitignored for privacy). |

---

## Summary Scorecard

| Test Case | Category | Status | Notes |
|---|---|---|---|
| TC-01: Out-of-Bounds Query | Boundary / Safety | ☐ Pass ☐ Fail | |
| TC-02: Factual Accuracy | Accuracy | ☐ Pass ☐ Fail | |
| TC-03: Multi-Document Retrieval | Retrieval Correctness | ☐ Pass ☐ Fail | |
| TC-04: Latency Benchmark | Performance | ☐ Pass ☐ Fail | Avg: ___ sec |
| TC-05: Edge Case Inputs | Robustness | ☐ Pass ☐ Fail | |
| TC-06: DB Persistence | Data Integrity | ☐ Pass ☐ Fail | |

---

## Troubleshooting Quick Reference

| Symptom | Likely Cause | Fix |
|---|---|---|
| `Connection refused` error | Ollama not running | Run `ollama serve` in a separate terminal |
| `model not found` error | Model not pulled | Run `ollama pull llama3` and `ollama pull nomic-embed-text` |
| Very slow responses (>60s) | CPU overloaded or model too large | Close other apps; try `llama3:8b-instruct-q4_K_M` |
| Answer ignores documents | System prompt too weak | Lower `temperature` to `0.0`; tighten system prompt |
| App crashes on large PDF | PDF parsing error or memory | Split PDF into smaller files; check for encrypted/scanned PDFs |
| Chunks not found | Embedding mismatch after DB rebuild | Click **Clear DB**, re-upload, and reprocess |
