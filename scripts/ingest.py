import argparse
import logging
import time

from app.chunker import chunk_documents
from app.config import configure_logging
from app.loaders import load_github_repo, load_pdf, load_url
from app.vectorstore import VectorStore

log = logging.getLogger(__name__)


def ingest(source: str) -> int:
    """Load → chunk → embed → store. Returns the number of chunks written."""
    start = time.time()
    log.info("Ingesting: %s", source)

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
        log.warning("No documents loaded from %s — check the source", source)
        return 0

    chunks = chunk_documents(docs)
    vs = VectorStore()

    # Delete-then-insert. Upsert alone never removes anything, so a file deleted
    # upstream — or a chunk index that shrank because a file got shorter — would
    # linger and keep being retrieved, letting the assistant answer from content
    # that no longer exists.
    removed = vs.delete_by_source(source)
    if removed:
        log.info("Removed %d stale chunks for %s before re-ingest", removed, source)

    added = vs.upsert(chunks)
    log.info("Ingested %d chunks in %.1fs — %d total in DB", added, time.time() - start, vs.count())
    return added


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="GitHub URL, web URL, or PDF path")
    args = parser.parse_args()
    ingest(args.source)
