import os
import argparse
from dotenv import load_dotenv
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()

# Ensure HF token is available under common env names
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token


def process_pdfs_to_chroma(
    files,
    persist_directory: str = "doc_vectorstore",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model=None,
):
    """Load one or more PDF files, split into chunks, embed and persist to Chroma.

    Args:
        files: list of file paths (or a single string path)
        persist_directory: folder to persist Chroma DB
        chunk_size: size of each text chunk
        chunk_overlap: overlap between chunks
        embedding_model: optional pre-instantiated embedding object
    Returns:
        number of documents/chunks added (int)
    """
    if isinstance(files, str):
        files = [files]

    working_dir = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(working_dir, persist_directory)
    os.makedirs(persist_dir, exist_ok=True)

    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings()

    all_texts = []
    for f in files:
        path = f if os.path.isabs(f) else os.path.join(working_dir, f)
        loader = PyPDFLoader(path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        all_texts.extend(chunks)

    if not all_texts:
        return 0

    vectordb = Chroma.from_documents(
        documents=all_texts, embedding=embedding_model, persist_directory=persist_dir
    )
    # Ensure DB is persisted
    try:
        vectordb.persist()
    except Exception:
        pass

    return len(all_texts)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Chroma vector store")
    parser.add_argument("--files", "-f", nargs="+", help="PDF file(s) to ingest")
    parser.add_argument("--dir", "-d", help="Directory with PDF files to ingest")
    parser.add_argument(
        "--persist", "-p", default="doc_vectorstore", help="Chroma persist directory"
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
        # If no files specified, default to ingesting all PDFs in the working directory
        working_dir = os.path.dirname(os.path.abspath(__file__))
        for fn in os.listdir(working_dir):
            if fn.lower().endswith(".pdf"):
                files.append(os.path.join(working_dir, fn))

    if not files:
        logging.error("No PDF files found in working directory and no args provided. Use --files or --dir to specify PDFs.")
        return

    count = process_pdfs_to_chroma(files, persist_directory=args.persist)
    logging.info(f"Processed {len(files)} file(s), created {count} chunks in Chroma.")


if __name__ == "__main__":
    main()
