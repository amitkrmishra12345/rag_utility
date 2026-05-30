import argparse
import logging

from dotenv import load_dotenv

from vectorstore_utility import (
    answer_with_vectorstore,
    load_vectorstore,
    search_vectorstore,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def search_docs(
    query: str,
    k: int = 4,
    persist_directory: str = "doc_vectorstore",
    vectordb=None,
):
    if vectordb is None:
        vectordb = load_vectorstore(persist_directory)
    return search_vectorstore(query, vectordb, k)


def answer_with_llm(
    query: str,
    k: int = 4,
    persist_directory: str = "doc_vectorstore",
    vectordb=None,
):
    if vectordb is None:
        vectordb = load_vectorstore(persist_directory)
    return answer_with_vectorstore(query, vectordb, k)


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
