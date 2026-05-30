import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("📚 PDF Q&A — Search and Answer")

st.subheader("1. Upload PDFs")
uploaded_files = st.file_uploader(
    "Upload PDF file(s) to index",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Index uploaded PDFs"):
    with st.spinner("Indexing PDFs..."):
        try:
            from doc_ingestion_utility import process_pdfs_to_vectorstore

            paths = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for uf in uploaded_files:
                    path = os.path.join(tmpdir, uf.name)
                    with open(path, "wb") as f:
                        f.write(uf.getbuffer())
                    paths.append(path)
                chunk_count = process_pdfs_to_vectorstore(
                    paths, persist_directory="doc_vectorstore"
                )
            st.success(f"Indexed {len(uploaded_files)} file(s), {chunk_count} chunk(s).")
        except Exception as e:
            st.error(f"Indexing failed: {e}")

st.divider()
st.subheader("2. Ask a question")

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_area("Enter your question or search query", height=120)
    k = st.number_input("Top-k passages to retrieve", min_value=1, max_value=20, value=4)
    persist = st.text_input("Vector store directory", value="doc_vectorstore")
    use_llm = st.checkbox("Use LLM to generate an answer (requires GROQ_API_KEY)")
    run = st.button("Run")

with col2:
    st.markdown("**Options**")
    st.markdown("- **Use LLM**: Groq answer from retrieved chunks")
    st.markdown("- **Search only**: top-k passages from the index")

if run:
    if not query or not query.strip():
        st.error("Please enter a query.")
    else:
        try:
            if use_llm:
                from user_query_retrieval_utility import answer_with_llm

                with st.spinner("Running LLM answer..."):
                    answer = answer_with_llm(query, k=k, persist_directory=persist)
                st.subheader("LLM Answer")
                st.markdown(answer)
            else:
                from user_query_retrieval_utility import search_docs

                with st.spinner("Searching documents..."):
                    hits = search_docs(query, k=k, persist_directory=persist)

                st.subheader(f"Top {len(hits)} hits")
                for i, h in enumerate(hits, start=1):
                    with st.expander(f"Hit {i}"):
                        st.write(h.get("page_content", ""))
                        st.markdown("**Metadata**")
                        st.json(h.get("metadata", {}))
        except Exception as e:
            st.error(f"Error: {e}")
