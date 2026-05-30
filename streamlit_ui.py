import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "doc_vectorstore"


def _upload_fingerprint(files) -> tuple:
    return tuple((f.name, f.size) for f in files)


def index_uploaded_pdfs(uploaded_files):
    """Index PDFs; returns (chunk_count, vectorstore)."""
    from doc_ingestion_utility import process_pdfs_to_vectorstore
    from vectorstore_utility import load_vectorstore

    paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for uf in uploaded_files:
            path = os.path.join(tmpdir, uf.name)
            with open(path, "wb") as f:
                f.write(uf.getbuffer())
            paths.append(path)
        result = process_pdfs_to_vectorstore(paths, persist_directory=PERSIST_DIR)

    if isinstance(result, tuple):
        return result[0], result[1]
    count = int(result)
    if count <= 0:
        return 0, None
    return count, load_vectorstore(PERSIST_DIR)


st.set_page_config(page_title="PDF Q&A", layout="wide")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.chunk_count = 0
    st.session_state.indexed_file_key = None

st.title("📚 PDF Q&A — Search and Answer")

st.subheader("1. Upload PDFs")
st.caption("PDFs are indexed automatically when you upload them.")

uploaded_files = st.file_uploader(
    "Upload PDF file(s)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    fingerprint = _upload_fingerprint(uploaded_files)
    if fingerprint != st.session_state.indexed_file_key:
        with st.spinner("Indexing PDFs automatically (first run may take a minute)..."):
            try:
                chunk_count, vectordb = index_uploaded_pdfs(uploaded_files)
                st.session_state.indexed_file_key = fingerprint

                if chunk_count == 0:
                    st.session_state.vectorstore = None
                    st.session_state.chunk_count = 0
                    st.warning(
                        "No text was extracted from the PDF(s). "
                        "Use text-based PDFs (not scanned images)."
                    )
                else:
                    st.session_state.vectorstore = vectordb
                    st.session_state.chunk_count = chunk_count
                    st.success(
                        f"Auto-indexed {len(uploaded_files)} file(s), "
                        f"{chunk_count} chunk(s). You can ask questions below."
                    )
            except Exception as e:
                st.session_state.vectorstore = None
                st.session_state.chunk_count = 0
                st.error(f"Auto-indexing failed: {e}")
else:
    if st.session_state.indexed_file_key is not None:
        st.session_state.indexed_file_key = None
        st.session_state.vectorstore = None
        st.session_state.chunk_count = 0

if st.session_state.vectorstore is None:
    try:
        from vectorstore_utility import load_vectorstore

        st.session_state.vectorstore = load_vectorstore(PERSIST_DIR)
        st.session_state.chunk_count = -1
    except FileNotFoundError:
        pass

if st.session_state.vectorstore is not None:
    chunks = st.session_state.chunk_count
    if chunks > 0:
        st.info(f"✅ Index ready ({chunks} chunks).")
    else:
        st.info("✅ Index loaded from disk.")
else:
    st.warning("⚠️ Upload PDF(s) above to build the index, then ask a question.")

st.divider()
st.subheader("2. Ask a question")

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_area("Enter your question or search query", height=120)
    k = st.number_input("Top-k passages to retrieve", min_value=1, max_value=20, value=4)
    use_llm = st.checkbox("Use LLM to generate an answer (requires GROQ_API_KEY)")
    run = st.button("Run", disabled=st.session_state.vectorstore is None)

with col2:
    st.markdown("**Options**")
    st.markdown("- **Use LLM**: Groq answer from retrieved chunks")
    st.markdown("- **Search only**: top-k passages from the index")

if run:
    if not query or not query.strip():
        st.error("Please enter a query.")
    else:
        try:
            db = st.session_state.vectorstore
            if use_llm:
                from user_query_retrieval_utility import answer_with_llm

                with st.spinner("Running LLM answer..."):
                    answer = answer_with_llm(
                        query, k=k, persist_directory=PERSIST_DIR, vectordb=db
                    )
                st.subheader("LLM Answer")
                st.markdown(answer)
            else:
                from user_query_retrieval_utility import search_docs

                with st.spinner("Searching documents..."):
                    hits = search_docs(
                        query, k=k, persist_directory=PERSIST_DIR, vectordb=db
                    )

                st.subheader(f"Top {len(hits)} hits")
                for i, h in enumerate(hits, start=1):
                    with st.expander(f"Hit {i}"):
                        st.write(h.get("page_content", ""))
                        st.markdown("**Metadata**")
                        st.json(h.get("metadata", {}))
        except Exception as e:
            st.error(f"Error: {e}")
