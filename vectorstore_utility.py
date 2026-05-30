"""FAISS-backed vector store (Streamlit Cloud–friendly; no Chroma/protobuf)."""

import os

from langchain_huggingface import HuggingFaceEmbeddings


def _persist_path(persist_directory: str = "doc_vectorstore") -> str:
    working_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(working_dir, persist_directory)


def _faiss_exists(persist_dir: str) -> bool:
    return os.path.isfile(os.path.join(persist_dir, "index.faiss")) and os.path.isfile(
        os.path.join(persist_dir, "index.pkl")
    )


def get_embeddings(embedding_model=None):
    if embedding_model is None:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        return HuggingFaceEmbeddings()
    return embedding_model


def add_documents(chunks, persist_directory: str = "doc_vectorstore", embedding_model=None):
    """Embed chunks and merge into on-disk FAISS index. Returns (chunk_count, FAISS db)."""
    from langchain_community.vectorstores import FAISS

    if not chunks:
        return 0, None

    persist_dir = _persist_path(persist_directory)
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = get_embeddings(embedding_model)

    if _faiss_exists(persist_dir):
        db = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(persist_dir)
    return len(chunks), db


def load_vectorstore(persist_directory: str = "doc_vectorstore", embedding_model=None):
    from langchain_community.vectorstores import FAISS

    persist_dir = _persist_path(persist_directory)
    if not _faiss_exists(persist_dir):
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. Upload and index PDFs first."
        )
    embeddings = get_embeddings(embedding_model)
    return FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _retrieve_documents(retriever, query: str):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query)
    raise AttributeError("Retriever does not support a known query API.")


def search_vectorstore(query: str, vectordb, k: int = 4):
    """Return top-k chunks from an in-memory or loaded FAISS index."""
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    results = _retrieve_documents(retriever, query)
    hits = []
    for r in results:
        hits.append(
            {
                "page_content": getattr(r, "page_content", str(r)),
                "metadata": getattr(r, "metadata", {}),
            }
        )
    return hits


def answer_with_vectorstore(query: str, vectordb, k: int = 4):
    """LLM answer using retrieved chunks from the given FAISS index."""
    from rag_utility import llm

    if llm is None:
        raise RuntimeError("LLM not configured; set GROQ_API_KEY in secrets.")

    from langchain_classic.chains.retrieval_qa.base import RetrievalQA

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    response = qa_chain.invoke({"query": query})
    return response.get("result")
