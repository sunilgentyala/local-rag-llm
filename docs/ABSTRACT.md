# Abstract

## Building a Privacy-First, Locally-Executed Retrieval-Augmented Generation System as a Cost-Effective Alternative to Large-Scale Model Fine-Tuning

---

### Overview

This project presents the design and implementation of a fully local, privacy-preserving question-answering system built on Retrieval-Augmented Generation (RAG). The system enables end-users to interact with their personal document collections through a natural-language interface without transmitting data to external servers, incurring per-query costs, or requiring specialized GPU hardware.

---

### Problem Statement

The rapid adoption of large language models (LLMs) in enterprise and personal productivity contexts has surfaced two opposing challenges:

1. **Privacy risk:** Mainstream LLM APIs (OpenAI, Anthropic, Google) require documents and queries to be sent to third-party infrastructure, creating inherent data-sovereignty concerns for sensitive material such as legal contracts, medical records, or proprietary research.

2. **Cost and complexity of customization:** Fine-tuning a base model on domain-specific data requires significant GPU compute, curated labeled datasets, and deep ML engineering expertise,resources unavailable to most individual users or small teams.

---

### Methodology

This system addresses both challenges through the following approach:

**Local inference engine:** [Ollama](https://ollama.com) is used to run quantized versions of open-source LLMs (`llama3`) and embedding models (`nomic-embed-text`) directly on consumer CPU hardware. No GPU is required.

**Retrieval-Augmented Generation (RAG):** Rather than fine-tuning the model's weights, the pipeline grounds the LLM's responses in relevant document context retrieved at query time. PDF documents are:

1. Parsed using `PyPDFLoader`
2. Split into overlapping chunks using `RecursiveCharacterTextSplitter` (1,000 tokens, 150-token overlap)
3. Embedded into vector representations using `nomic-embed-text`
4. Stored in a persistent local `ChromaDB` vector database
5. Retrieved by cosine similarity at query time (Top-K = 4 chunks)
6. Passed as grounding context into `llama3` via a structured system prompt

**Hallucination mitigation:** The system prompt explicitly instructs the model to answer exclusively from the retrieved context and to decline questions it cannot answer from the documents, reducing confabulation.

**User interface:** A [Streamlit](https://streamlit.io) web application provides a browser-based chat interface accessible at `localhost:8501`, requiring no frontend engineering expertise to operate.

---

### Implementation

| Component | Technology | Role |
|---|---|---|
| Inference Runtime | Ollama | Local LLM and embedding server |
| Language Model | llama3 (8B) | Answer generation |
| Embedding Model | nomic-embed-text | Semantic vector representation |
| Orchestration | LangChain 0.3 (LCEL) | RAG chain assembly |
| Vector Database | ChromaDB (local) | Persistent semantic storage |
| User Interface | Streamlit | Browser-based chat UI |
| Document Parser | PyPDFLoader | PDF text extraction |

---

### Expected Outcomes

- **Zero marginal cost:** No API usage fees regardless of query volume.
- **Complete data sovereignty:** Documents and query history remain exclusively on the user's local machine.
- **Acceptable latency on CPU:** Typical response latency of 5–30 seconds on a modern laptop CPU, depending on document size and system resources.
- **Domain adaptability:** The knowledge base can be updated instantly by uploading new PDFs — no model retraining required.

---

### Future Work

A supervised fine-tuning roadmap using QLoRA (Quantized Low-Rank Adaptation) via [Unsloth](https://github.com/unslothai/unsloth) on Google Colab is outlined in [`docs/RESEARCH.md`](RESEARCH.md). This path becomes viable when labeled domain-specific datasets are available and the user's accuracy requirements exceed what retrieval-based grounding can achieve.

---

### Conclusion

This project demonstrates that privacy-preserving, cost-free document intelligence is achievable today on consumer hardware, without fine-tuning, using open-source tooling. RAG on local LLMs represents a pragmatic, production-viable alternative to cloud-based AI services for privacy-sensitive use cases.
