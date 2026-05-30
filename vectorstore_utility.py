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


def add_documents(chunks, persist_directory: str = "doc_vectorstore", embedding_model=None) -> int:
    """Embed chunks and merge into on-disk FAISS index. Returns chunk count added."""
    from langchain_community.vectorstores import FAISS

    if not chunks:
        return 0

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
    return len(chunks)


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
