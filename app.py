"""
Local RAG LLM — Enhanced Edition
==================================
New in this version:
  - Multi-format documents  : PDF, DOCX, TXT, CSV
  - Conversation memory     : last N turns passed to LLM
  - Web search (opt-in)     : DuckDuckGo — no API key needed
  - Hybrid mode             : local docs + web, combined context
  - SQLite chat history     : sessions survive restarts
  - Document generator      : IEEE, Elsevier, Blog DOCX export

Stack   : Ollama · llama3 · nomic-embed-text · LangChain · ChromaDB · Streamlit
Privacy : 100% local. Web search is opt-in and clearly labelled.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import db
from generators import generate_blog_docx, generate_elsevier_docx, generate_ieee_docx


load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_PATH     = ".chroma"
COLLECTION_NAME = "rag_docs"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE      = 1_000
CHUNK_OVERLAP   = 150
TOP_K_DOCS      = 4
TOP_K_GENERATE  = 6
TOP_K_FETCH     = 20    # candidates pulled before reranking
MEMORY_TURNS    = 5

# ── Prompts ───────────────────────────────────────────────────────────────────
_DOCS_PROMPT = """\
You are a precise research assistant. Answer ONLY using the document context below.
If the answer cannot be found, respond: "I could not find an answer in the uploaded documents."
Never fabricate facts or draw on outside knowledge.

Prior conversation:
{history}

Document context:
{context}"""

_WEB_PROMPT = """\
You are a helpful assistant. Answer the question using the web search results below.
Cite sources by URL where relevant. If results are insufficient, say so clearly.

Web results:
{context}"""

_HYBRID_PROMPT = """\
You are a research assistant with access to both uploaded documents and live web search.
Answer using all context below. Clearly indicate whether a fact comes from a document or the web.

Prior conversation:
{history}

Context (documents + web):
{context}"""

_SECTION_PROMPT = """\
You are an expert academic writer. Write a {section} section for a {paper_type}.
Base your writing strictly on the context — do not invent citations or statistics.
Use formal, precise language. Target approximately {word_count} words.

Source context:
{context}

Write the {section} section now:"""

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local RAG LLM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

db.init_db()

for _k, _v in {
    "messages":       [],
    "vector_store":   None,
    "all_chunks":     [],
    "docs_processed": False,
    "doc_count":      0,
    "search_mode":    "Documents",
    "session_id":     None,
}.items():
    st.session_state.setdefault(_k, _v)


# ── Model helpers ─────────────────────────────────────────────────────────────
def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

def _llm(temperature: float = 0.1) -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL, base_url=OLLAMA_BASE_URL,
        temperature=temperature, streaming=True,
    )


@st.cache_resource
def _reranker():
    """Download and cache the cross-encoder reranker model once per server session."""
    try:
        from flashrank import Ranker
        return Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".flashrank_cache")
    except Exception:
        return None


# ── Document loading ──────────────────────────────────────────────────────────
_LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc":  Docx2txtLoader,
    ".csv":  CSVLoader,
}

def load_documents(files: list) -> list:
    all_docs = []
    for uf in files:
        suffix = Path(uf.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            path = tmp.name
        try:
            loader_cls = _LOADER_MAP.get(suffix, TextLoader)
            kwargs = {"encoding": "utf-8"} if loader_cls is TextLoader else {}
            pages  = loader_cls(path, **kwargs).load()
            for p in pages:
                p.metadata["source"] = uf.name
            all_docs.extend(pages)
        except Exception as e:
            st.sidebar.warning(f"Could not load {uf.name}: {e}")
        finally:
            os.unlink(path)
    return all_docs


# ── Vector store ──────────────────────────────────────────────────────────────
def _chunk(docs: list) -> list:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents(docs)

def _build_store(chunks: list) -> Chroma:
    return Chroma.from_documents(
        documents=chunks, embedding=_embeddings(),
        persist_directory=CHROMA_PATH, collection_name=COLLECTION_NAME,
    )

def _load_store() -> Chroma | None:
    if not Path(CHROMA_PATH).exists():
        return None
    try:
        s = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=_embeddings(),
            collection_name=COLLECTION_NAME,
        )
        return s if s._collection.count() > 0 else None
    except Exception:
        return None

def _load_store_and_chunks() -> tuple[Chroma | None, list]:
    """Like _load_store, but also reconstructs the chunk list BM25 needs."""
    store = _load_store()
    if store is None:
        return None, []
    result = store._collection.get(include=["documents", "metadatas"])
    chunks = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(result["documents"], result["metadatas"])
    ]
    return store, chunks

def _clear_store() -> None:
    import shutil
    if Path(CHROMA_PATH).exists():
        shutil.rmtree(CHROMA_PATH)


# ── Hybrid retrieval + reranking ─────────────────────────────────────────────
def _hybrid_retriever(store: Chroma, chunks: list) -> EnsembleRetriever:
    """Combine BM25 keyword search with ChromaDB vector search (50 / 50)."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K_FETCH
    vec = store.as_retriever(search_kwargs={"k": TOP_K_FETCH})
    return EnsembleRetriever(retrievers=[bm25, vec], weights=[0.5, 0.5])

def _rerank(query: str, docs: list, top_k: int) -> list:
    """Cross-encoder rerank; falls back to the first top_k candidates if unavailable."""
    ranker = _reranker()
    if ranker is None or not docs:
        return docs[:top_k]
    try:
        from flashrank import RerankRequest
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        results  = ranker.rerank(RerankRequest(query=query, passages=passages))
        return [docs[r["id"]] for r in results[:top_k]]
    except Exception:
        return docs[:top_k]

def _retrieve(query: str, store: Chroma, chunks: list, top_k: int) -> list:
    """Hybrid BM25 + vector retrieval, reranked down to top_k. Falls back to
    plain similarity search if there are no chunks available for BM25."""
    if not chunks:
        return store.as_retriever(search_kwargs={"k": top_k}).invoke(query)
    candidates = _hybrid_retriever(store, chunks).invoke(query)
    return _rerank(query, candidates, top_k)


# ── Web search ────────────────────────────────────────────────────────────────
def web_search(query: str, n: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=n))
        if not results:
            return "No web results found."
        return "\n\n---\n\n".join(
            f"**{r.get('title','')}**\n{r.get('body','')}\nURL: {r.get('href','')}"
            for r in results
        )
    except Exception as e:
        return f"Web search unavailable: {e}"


# ── Conversation memory ───────────────────────────────────────────────────────
def fmt_history(msgs: list, n: int = MEMORY_TURNS) -> str:
    recent = msgs[-(n * 2):]
    if not recent:
        return "None."
    return "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:400]}"
        for m in recent
    )


# ── Answer function ───────────────────────────────────────────────────────────
def answer(user_input: str, mode: str, history: str,
           store: Chroma | None) -> dict:
    llm = _llm()

    if mode == "Web Search":
        ctx = web_search(user_input)
        ans = (
            ChatPromptTemplate.from_messages([
                ("system", _WEB_PROMPT), ("human", "{input}"),
            ])
            | llm | StrOutputParser()
        ).invoke({"context": ctx, "input": user_input})
        return {"answer": ans, "doc_sources": [], "web_context": ctx}

    # Documents or Hybrid — need a store
    doc_docs = _retrieve(user_input, store, st.session_state.all_chunks, TOP_K_DOCS)
    doc_ctx  = "\n\n".join(d.page_content for d in doc_docs)

    if mode == "Hybrid":
        web_ctx  = web_search(user_input, n=3)
        full_ctx = f"[DOCUMENTS]\n{doc_ctx}\n\n[WEB]\n{web_ctx}"
        ans = (
            ChatPromptTemplate.from_messages([
                ("system", _HYBRID_PROMPT), ("human", "{input}"),
            ])
            | llm | StrOutputParser()
        ).invoke({"context": full_ctx, "history": history, "input": user_input})
        return {"answer": ans, "doc_sources": doc_docs, "web_context": web_ctx}

    # Documents only
    ans = (
        ChatPromptTemplate.from_messages([
            ("system", _DOCS_PROMPT), ("human", "{input}"),
        ])
        | llm | StrOutputParser()
    ).invoke({"context": doc_ctx, "history": history, "input": user_input})
    return {"answer": ans, "doc_sources": doc_docs, "web_context": ""}


def fmt_sources(docs: list) -> str:
    seen, lines = set(), []
    for d in docs:
        key = f"{d.metadata.get('source')}::{d.metadata.get('page', '?')}"
        if key not in seen:
            seen.add(key)
            lines.append(
                f"- **{d.metadata.get('source', '?')}** · "
                f"page {d.metadata.get('page', '?')}"
            )
    return "\n".join(lines)


# ── Section generation via RAG ────────────────────────────────────────────────
def generate_section(section: str, paper_type: str, wc: int, store: Chroma) -> str:
    docs = _retrieve(section, store, st.session_state.all_chunks, TOP_K_GENERATE)
    ctx  = "\n\n".join(d.page_content for d in docs)
    return (
        ChatPromptTemplate.from_messages([
            ("system", _SECTION_PROMPT), ("human", "Write the section now."),
        ])
        | _llm(temperature=0.3)
        | StrOutputParser()
    ).invoke({
        "section": section, "paper_type": paper_type,
        "context": ctx, "word_count": wc,
    })


# ── DB session helper ─────────────────────────────────────────────────────────
def _ensure_session(mode: str) -> int:
    if not st.session_state.session_id:
        st.session_state.session_id = db.new_session(
            f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}", mode
        )
    return st.session_state.session_id


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🧠 Local RAG LLM")
    st.caption("100% private · powered by Ollama")
    st.divider()

    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF · DOCX · TXT · CSV",
        type=["pdf", "docx", "doc", "txt", "csv"],
        accept_multiple_files=True,
    )

    c1, c2 = st.columns(2)
    process_btn = c1.button("⚙️ Process", type="primary", use_container_width=True)
    clear_btn   = c2.button("🗑️ Clear DB", use_container_width=True)

    st.divider()
    st.subheader("🔍 Search Mode")
    mode = st.radio(
        "Answer from:",
        ["Documents", "Web Search", "Hybrid"],
        help=(
            "Documents — RAG from your files only.\n\n"
            "Web Search — DuckDuckGo (requires internet, no API key).\n\n"
            "Hybrid — local docs + web combined."
        ),
    )
    st.session_state.search_mode = mode

    st.divider()
    st.subheader("⚙️ Config")
    st.markdown(f"""\
| | |
|---|---|
| LLM | `{LLM_MODEL}` |
| Embeddings | `{EMBED_MODEL}` |
| Memory | last `{MEMORY_TURNS}` turns |
| Top-K | `{TOP_K_DOCS}` |
""")
    st.divider()

    # Knowledge-base status
    if st.session_state.docs_processed:
        st.success(f"✅ KB ready · {st.session_state.doc_count} chunks")
    else:
        existing, existing_chunks = _load_store_and_chunks()
        if existing:
            st.info(f"💾 Existing KB ({len(existing_chunks)} chunks)")
            if not st.session_state.vector_store:
                st.session_state.vector_store   = existing
                st.session_state.all_chunks     = existing_chunks
                st.session_state.docs_processed = True
                st.session_state.doc_count      = len(existing_chunks)
        elif mode == "Documents":
            st.warning("⚠️ No documents indexed.")
        else:
            st.info(f"🌐 {mode} active — no documents needed.")


# ── Process ───────────────────────────────────────────────────────────────────
if process_btn:
    if not uploaded_files:
        st.sidebar.error("Upload at least one file first.")
    else:
        with st.spinner("Loading and chunking…"):
            raw    = load_documents(uploaded_files)
            chunks = _chunk(raw)
        with st.spinner(f"Embedding {len(chunks)} chunks — first run may take a minute…"):
            store = _build_store(chunks)

        st.session_state.vector_store   = store
        st.session_state.all_chunks     = chunks
        st.session_state.docs_processed = True
        st.session_state.doc_count      = len(chunks)
        st.session_state.messages       = []

        sid = _ensure_session(mode)
        for uf in uploaded_files:
            db.save_document(sid, uf.name, Path(uf.name).suffix.lower(), len(chunks))

        st.sidebar.success(f"✅ {len(chunks)} chunks from {len(uploaded_files)} file(s)")
        st.rerun()

# ── Clear ─────────────────────────────────────────────────────────────────────
if clear_btn:
    _clear_store()
    st.session_state.update({
        "vector_store":   None,
        "all_chunks":     [],
        "docs_processed": False,
        "doc_count":      0,
        "messages":       [],
        "session_id":     None,
    })
    st.sidebar.success("Cleared.")
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_gen, tab_hist = st.tabs(["💬 Chat", "📝 Generate", "📜 History"])


# ───────────────────────────────────────────────────────────── Chat tab ───────
with tab_chat:
    _mode  = st.session_state.search_mode
    _store = st.session_state.vector_store

    _need_docs = _mode in ("Documents", "Hybrid") and not st.session_state.docs_processed

    if _need_docs:
        st.info(
            "👈 Upload documents in the sidebar and click **Process** to start.\n\n"
            "Switch to **Web Search** mode if you want to chat without uploading files."
        )
    else:
        _badge = {
            "Documents":  "📄 Documents mode — answers from your uploaded files",
            "Web Search": "🌐 Web Search mode — answers from DuckDuckGo (internet required)",
            "Hybrid":     "🔀 Hybrid mode — local documents + web combined",
        }[_mode]
        st.caption(_badge)

        # Render message history
        for _msg in st.session_state.messages:
            _role = "user" if isinstance(_msg, HumanMessage) else "assistant"
            with st.chat_message(_role, avatar="🧑" if _role == "user" else "🤖"):
                st.markdown(_msg.content)

        if _user_input := st.chat_input("Ask a question…"):
            _sid  = _ensure_session(_mode)
            _hist = fmt_history(st.session_state.messages)

            st.session_state.messages.append(HumanMessage(content=_user_input))
            db.save_message(_sid, "user", _user_input)

            with st.chat_message("user", avatar="🧑"):
                st.markdown(_user_input)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking…"):
                    try:
                        result   = answer(_user_input, _mode, _hist, _store)
                        _answer  = result["answer"]

                        st.markdown(_answer)

                        if result["doc_sources"]:
                            with st.expander("📚 Document Sources"):
                                st.markdown(fmt_sources(result["doc_sources"]))
                        if result.get("web_context"):
                            with st.expander("🌐 Web Sources"):
                                _wc = result["web_context"]
                                st.markdown(_wc[:3000] + ("…" if len(_wc) > 3000 else ""))

                        st.session_state.messages.append(AIMessage(content=_answer))
                        db.save_message(_sid, "assistant", _answer)

                    except Exception as _e:
                        st.error(
                            f"⚠️ `{_e}`\n\n"
                            f"Make sure Ollama is running (`ollama serve`) "
                            f"and both `{LLM_MODEL}` and `{EMBED_MODEL}` are pulled."
                        )


# ─────────────────────────────────────────────────────────── Generate tab ─────
with tab_gen:
    st.header("Generate Document")
    st.caption("Uses your indexed documents as source material via RAG.")

    if not st.session_state.docs_processed:
        st.info("📄 Upload and process documents first — the generator reads from them.")
    else:
        _doc_type = st.selectbox(
            "Format",
            ["IEEE Conference Paper", "Elsevier Journal Article", "Blog Post"],
        )

        _cl, _cr = st.columns(2)
        with _cl:
            _title       = st.text_input("Title")
            _authors     = st.text_input("Author(s)", placeholder="First Last, First Last")
            _affiliation = st.text_input("Affiliation", placeholder="University / Company, Country")
        with _cr:
            _keywords   = st.text_input("Keywords / Tags (comma-separated)")
            _highlights = ""
            if _doc_type == "Elsevier Journal Article":
                _highlights = st.text_area(
                    "Highlights (one per line, max 5)",
                    placeholder="Novel approach to X\nAchieves Y% improvement\n...",
                    height=110,
                )

        st.divider()

        _default_sections = {
            "IEEE Conference Paper":    [
                "Abstract", "Introduction", "Related Work",
                "Methodology", "Results", "Conclusion",
            ],
            "Elsevier Journal Article": [
                "Abstract", "Introduction", "Literature Review",
                "Methodology", "Results", "Discussion", "Conclusion",
            ],
            "Blog Post": [
                "Introduction", "Main Content", "Key Takeaways", "Conclusion",
            ],
        }[_doc_type]

        _selected = st.multiselect(
            "Sections to generate", _default_sections, default=_default_sections
        )
        _custom = st.text_input("Custom section (optional)")
        if _custom.strip():
            _selected.append(_custom.strip())

        _wc = st.slider("Approx. words per section", 100, 500, 200, step=50)

        if st.button("🚀 Generate", type="primary", use_container_width=True):
            if not _title:
                st.error("Enter a title.")
            elif not _selected:
                st.error("Select at least one section.")
            else:
                _gen_store = _load_store()
                if not _gen_store:
                    st.error("No indexed documents found. Process files first.")
                else:
                    _generated = {}
                    _prog = st.progress(0)
                    for _i, _sec in enumerate(_selected):
                        _prog.progress((_i + 1) / len(_selected), text=f"Generating: {_sec}…")
                        _generated[_sec] = generate_section(_sec, _doc_type, _wc, _gen_store)
                    _prog.empty()
                    st.success("Done!")

                    with st.expander("📄 Preview", expanded=True):
                        for _name, _content in _generated.items():
                            st.markdown(f"### {_name}")
                            st.markdown(_content)
                            st.divider()

                    # Abstract is always a named param; keep all other sections
                    # in the dict so IEEE/Elsevier number them correctly.
                    # Blog is the exception: it has named intro/conclusion slots.
                    _abstract = _generated.pop("Abstract", "")

                    if _doc_type == "IEEE Conference Paper":
                        _data     = generate_ieee_docx(
                            _title, _authors, _affiliation,
                            _abstract, _keywords, _generated,
                        )
                        _filename = f"{_title[:40].replace(' ', '_')}_IEEE.docx"

                    elif _doc_type == "Elsevier Journal Article":
                        _data     = generate_elsevier_docx(
                            _title, _authors, _affiliation,
                            _abstract, _keywords, _generated, _highlights,
                        )
                        _filename = f"{_title[:40].replace(' ', '_')}_Elsevier.docx"

                    else:
                        _intro      = _generated.pop("Introduction", "")
                        _conclusion = _generated.pop("Conclusion", "")
                        _data     = generate_blog_docx(
                            _title, _authors, _intro, _generated, _conclusion, _keywords
                        )
                        _filename = f"{_title[:40].replace(' ', '_')}_Blog.docx"

                    st.download_button(
                        "⬇️ Download DOCX",
                        data=_data,
                        file_name=_filename,
                        mime="application/vnd.openxmlformats-officedocument"
                             ".wordprocessingml.document",
                        use_container_width=True,
                    )


# ──────────────────────────────────────────────────────────── History tab ─────
with tab_hist:
    st.header("Chat History")
    st.caption("All sessions saved locally in SQLite — nothing leaves your machine.")

    if st.button("🆕 New Session"):
        st.session_state.update({"session_id": None, "messages": []})
        st.rerun()

    _sessions = db.get_sessions()
    if not _sessions:
        st.info("No sessions yet. Start chatting to create history.")
    else:
        for _sess in _sessions:
            _label = (
                f"📌 {_sess['name']}  ·  "
                f"{_sess['mode']}  ·  "
                f"{_sess['created'][:16]}"
            )
            with st.expander(_label):
                _docs = db.get_session_documents(_sess["id"])
                if _docs:
                    st.markdown("**Files processed:**")
                    for _d in _docs:
                        st.markdown(
                            f"- `{_d['filename']}` "
                            f"({_d['filetype']}, {_d['chunks']} chunks)"
                        )
                    st.divider()

                _msgs = db.get_messages(_sess["id"])
                if _msgs:
                    for _m in _msgs:
                        _r = "🧑 **User**" if _m["role"] == "user" else "🤖 **Assistant**"
                        st.markdown(f"{_r} · _{_m['ts'][:16]}_\n\n{_m['content']}")
                        st.markdown("---")
                else:
                    st.info("No messages in this session.")

                if st.button("🗑️ Delete session", key=f"del_{_sess['id']}"):
                    db.delete_session(_sess["id"])
                    st.rerun()


st.divider()
st.caption(
    "🔒 Local RAG LLM Enhanced · "
    "Ollama · LangChain · ChromaDB · Streamlit · SQLite"
)
