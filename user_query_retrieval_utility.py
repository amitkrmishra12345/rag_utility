import os
from dotenv import load_dotenv
import argparse
import logging

from vectorstore_utility import load_vectorstore

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_retriever(persist_directory: str = "doc_vectorstore", k: int = 4):
    vectordb = load_vectorstore(persist_directory)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever, vectordb


def _retrieve_documents(retriever, query: str):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query)
    raise AttributeError("Retriever does not support a known query API.")


def search_docs(query: str, k: int = 4, persist_directory: str = "doc_vectorstore"):
    """Return top-k document chunks from the FAISS index."""
    retriever, _ = _load_retriever(persist_directory, k)
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


def answer_with_llm(query: str, k: int = 4, persist_directory: str = "doc_vectorstore"):
    try:
        from rag_utility import llm
    except Exception as e:
        raise RuntimeError("Could not import LLM from rag_utility: %s" % e)

    if llm is None:
        raise RuntimeError("LLM not configured; set GROQ_API_KEY or provide an LLM.")

    from langchain_classic.chains.retrieval_qa.base import RetrievalQA

    retriever, _ = _load_retriever(persist_directory, k)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    response = qa_chain.invoke({"query": query})
    return response.get("result")


def main():
    parser = argparse.ArgumentParser(description="Query vector store or ask via LLM")
    parser.add_argument("query", help="Query text to search")
    parser.add_argument("--k", type=int, default=4, help="Top-k passages to retrieve")
    parser.add_argument("--persist", default="doc_vectorstore", help="Vector store directory")
    parser.add_argument(
        "--llm", action="store_true", help="Use LLM to produce an answer from retrieved docs"
    )
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
