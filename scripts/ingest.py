import argparse
import time

from app.chunker import chunk_documents
from app.loaders import load_github_repo, load_pdf, load_url
from app.vectorstore import VectorStore


def ingest(source: str):
    start = time.time()
    print(f"\n{'='*50}")
    print(f"Ingesting: {source}")

    # Auto-detect source type
    if source.startswith("https://github.com"):
        docs = load_github_repo(source)
    elif source.startswith("http"):
        docs = load_url(source)
    elif source.endswith(".pdf"):
        docs = load_pdf(source)
    else:
        raise ValueError(f"Unknown source type: {source}")

    if not docs:
        print("No documents loaded — check the source URL")
        return

    chunks = chunk_documents(docs)
    vs = VectorStore()
    added = vs.upsert(chunks)

    elapsed = time.time() - start
    print(f"\n✓ Ingested {added} chunks in {elapsed:.1f}s")
    print(f"✓ Total in DB: {vs.count()} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="GitHub URL, web URL, or PDF path")
    args = parser.parse_args()
    ingest(args.source)
