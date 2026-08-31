"""
DocuChat - Streamlit frontend.

Run with:  streamlit run app.py
"""
import streamlit as st

from src import config
from src.ingest import load_documents, chunk_documents, build_index
from src.retriever import Retriever
from src.pipeline import answer_question

st.set_page_config(page_title="DocuChat", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_retriever():
    """Cached so the embedding + cross-encoder models load only once per session."""
    return Retriever()


def index_exists() -> bool:
    return config.FAISS_INDEX_PATH.exists() and config.METADATA_PATH.exists()


# --- Sidebar: document upload + indexing ---------------------------------
with st.sidebar:
    st.header("📁 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or notes (.pdf, .txt, .md)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for uf in uploaded_files:
            dest = config.RAW_DOCS_DIR / uf.name
            dest.write_bytes(uf.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to {config.RAW_DOCS_DIR}")

    existing = list(config.RAW_DOCS_DIR.glob("*"))
    existing = [f for f in existing if f.name != ".gitkeep"]
    st.caption(f"{len(existing)} document(s) currently in data/raw_docs/")
    for f in existing:
        st.text(f"• {f.name}")

    if st.button("🔨 Build / Rebuild Index", use_container_width=True):
        with st.spinner("Loading documents, chunking, and embedding... this may take a minute."):
            try:
                docs = load_documents()
                chunks = chunk_documents(docs)
                build_index(chunks)
                st.cache_resource.clear()  # force retriever reload with new index
                st.success(f"Index built from {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    st.divider()
    st.caption(
        "Pipeline: Sentence-Transformer embeddings → FAISS semantic search "
        "→ cross-encoder rerank → Gemini generation, grounded in your own documents."
    )

# --- Main: chat interface -------------------------------------------------
st.title("📄 DocuChat")
st.caption("Ask questions about your own documents, answered with cited sources.")

if not index_exists():
    st.info("No index yet. Upload documents and click **Build / Rebuild Index** in the sidebar to get started.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, Answer)

question = st.chat_input("Ask a question about your documents...")

if question:
    try:
        retriever = get_retriever()
        with st.spinner("Retrieving relevant passages and generating an answer..."):
            result = answer_question(retriever, question)
        st.session_state.history.append((question, result))
    except Exception as e:
        st.error(f"Something went wrong: {e}")

for q, result in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(result.text)
        if result.passages:
            with st.expander(f"📚 Sources ({len(result.passages)} passages)"):
                for p in result.passages:
                    location = f"{p.source}" + (f", page {p.page}" if p.page else "")
                    st.markdown(f"**{location}**  \u2014 relevance score: `{p.score:.3f}`")
                    st.text(p.text[:500] + ("..." if len(p.text) > 500 else ""))
                    st.divider()
