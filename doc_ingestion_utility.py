import os
import argparse
from dotenv import load_dotenv
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorstore_utility import add_documents, load_vectorstore

load_dotenv()


def _coerce_add_result(result, persist_directory: str, embedding_model=None):
    """Support (count, db) or legacy int-only return from add_documents."""
    if isinstance(result, tuple):
        return result[0], result[1]
    count = int(result)
    if count <= 0:
        return 0, None
    return count, load_vectorstore(persist_directory, embedding_model)

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token


def process_pdfs_to_vectorstore(
    files,
    persist_directory: str = "doc_vectorstore",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model=None,
):
    """Load PDFs, chunk, embed, and persist to a FAISS index."""
    if isinstance(files, str):
        files = [files]

    working_dir = os.path.dirname(os.path.abspath(__file__))
    all_texts = []
    for f in files:
        path = f if os.path.isabs(f) else os.path.join(working_dir, f)
        loader = PyPDFLoader(path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        all_texts.extend(splitter.split_documents(docs))

    if not all_texts:
        return 0, None

    result = add_documents(all_texts, persist_directory, embedding_model)
    return _coerce_add_result(result, persist_directory, embedding_model)


# Backward-compatible name used by older streamlit_ui copies
process_pdfs_to_chroma = process_pdfs_to_vectorstore


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS vector store")
    parser.add_argument("--files", "-f", nargs="+", help="PDF file(s) to ingest")
    parser.add_argument("--dir", "-d", help="Directory with PDF files to ingest")
    parser.add_argument(
        "--persist", "-p", default="doc_vectorstore", help="Vector store directory"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    files = []
    if args.dir:
        dirpath = args.dir if os.path.isabs(args.dir) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.dir
        )
        for fn in os.listdir(dirpath):
            if fn.lower().endswith(".pdf"):
                files.append(os.path.join(dirpath, fn))
    if args.files:
        files.extend(args.files)

    if not files:
        working_dir = os.path.dirname(os.path.abspath(__file__))
        for fn in os.listdir(working_dir):
            if fn.lower().endswith(".pdf"):
                files.append(os.path.join(working_dir, fn))

    if not files:
        logging.error(
            "No PDF files found. Use --files or --dir to specify PDFs."
        )
        return

    count, _ = process_pdfs_to_vectorstore(files, persist_directory=args.persist)
    logging.info("Processed %s file(s), %s chunk(s) in vector store.", len(files), count)


if __name__ == "__main__":
    main()
