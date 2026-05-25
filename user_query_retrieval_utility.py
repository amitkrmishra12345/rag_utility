import os
from dotenv import load_dotenv
import argparse
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Helper: load Chroma and return retriever and embedding
def _load_retriever(persist_directory: str = "doc_vectorstore", k: int = 4):
    working_dir = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(working_dir, persist_directory)
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(f"Chroma persist directory not found: {persist_dir}")

    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever, embedding, vectordb


def _retrieve_documents(retriever, query: str):
    # Debug: print available methods for this retriever
    print("Retriever type:", type(retriever))
    print("Retriever dir:", dir(retriever))
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query)
    if hasattr(retriever, "get_relevant_texts"):
        texts = retriever.get_relevant_texts(query)
        return [type("Doc", (), {"page_content": t, "metadata": {}})() for t in texts]
    # Fallback: use private _get_relevant_documents if available, with run_manager=None
    if hasattr(retriever, "_get_relevant_documents"):
        return retriever._get_relevant_documents(query, run_manager=None)
    raise AttributeError(
        f"Retriever object has no supported retrieval method (get_relevant_documents/retrieve/get_relevant_texts/_get_relevant_documents). Available: {dir(retriever)}"
    )


def search_docs(query: str, k: int = 4, persist_directory: str = "doc_vectorstore"):
    """Return top-k document chunks (texts + metadata) from Chroma for a query."""
    retriever, _, _ = _load_retriever(persist_directory, k)
    results = _retrieve_documents(retriever, query)
    hits = []
    for r in results:
        hits.append({"page_content": getattr(r, "page_content", str(r)), "metadata": getattr(r, "metadata", {})})
    return hits


def answer_with_llm(query: str, k: int = 4, persist_directory: str = "doc_vectorstore"):
    """Use configured LLM (if available) to answer a query using RetrievalQA.

    This will raise a RuntimeError if no LLM is configured in the environment (see `rag_utility.py`).
    """
    # Lazy import rag_utility to reuse its LLM and embedding setup
    try:
        from rag_utility import llm, embedding
    except Exception as e:
        raise RuntimeError("Could not import LLM from rag_utility: %s" % e)

    if llm is None:
        raise RuntimeError("LLM not configured; set GROQ_API_KEY or provide an LLM.")

    retriever, _, _ = _load_retriever(persist_directory, k)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    response = qa_chain.invoke({"query": query})
    return response.get("result")


def main():
    parser = argparse.ArgumentParser(description="Query Chroma vectorstore or ask via LLM")
    parser.add_argument("query", help="Query text to search")
    parser.add_argument("--k", type=int, default=4, help="Top-k passages to retrieve")
    parser.add_argument("--persist", default="doc_vectorstore", help="Chroma persist directory")
    parser.add_argument("--llm", action="store_true", help="Use LLM to produce an answer from retrieved docs")
    args = parser.parse_args()

    if args.llm:
        try:
            answer = answer_with_llm(args.query, k=args.k, persist_directory=args.persist)
            print("LLM Answer:\n", answer)
        except Exception as e:
            logging.error("LLM answer failed: %s", e)
    else:
        hits = search_docs(args.query, k=args.k, persist_directory=args.persist)
        for i, h in enumerate(hits, 1):
            print(f"--- Hit {i} ---")
            print(h.get("page_content", ""))
            print("metadata:", h.get("metadata", {}))


if __name__ == "__main__":
    main()
