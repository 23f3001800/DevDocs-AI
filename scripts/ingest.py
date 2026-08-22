import argparse
import logging
import time

from app.chunker import chunk_documents
from app.config import configure_logging
from app.loaders import load_github_repo, load_pdf, load_url
from app.vectorstore import VectorStore

log = logging.getLogger(__name__)


def ingest(source: str, owner: str | None = None) -> dict:
    """Load → chunk → embed → store.

    Returns {"chunks_added": int, "kind": str, "commit_sha": str} — the caller
    (app.main's ingest job) uses this to record a `sources` row.

    WHY `owner`? Retrieval is per-user private. Every chunk from this ingest
    is tagged metadata["owner"] = owner so it is only ever retrievable by
    that user (or an admin, who searches across every owner). `owner=None`
    (the CLI's default) leaves chunks untagged — untagged chunks are visible
    to nobody under the per-user filter, so CLI ingests without an explicit
    owner are effectively admin-only debugging aids, not user-facing content.
    """
    start = time.time()
    log.info("Ingesting: %s (owner=%s)", source, owner or "-")

    # Auto-detect source type. Repos are version-tagged and non-destructive
    # (see below); URLs and PDFs are not, and stay delete-then-insert.
    is_repo = source.startswith("https://github.com")
    if is_repo:
        docs = load_github_repo(source)
        kind = "github"
    elif source.startswith("http"):
        docs = load_url(source)
        kind = "url"
    elif source.endswith(".pdf"):
        docs = load_pdf(source)
        kind = "pdf"
    else:
        raise ValueError(f"Unknown source type: {source}")

    if not docs:
        log.warning("No documents loaded from %s — check the source", source)
        return {"chunks_added": 0, "kind": kind, "commit_sha": ""}

    # Repo docs already carry commit_sha from load_github_repo; URL/PDF docs
    # don't have one, which is what keeps their chunk ids unchanged (see
    # vectorstore.make_chunk_id) and their delete-then-insert behaviour below.
    commit_sha = docs[0].metadata.get("commit_sha", "") if is_repo else ""

    if owner:
        for d in docs:
            d.metadata["owner"] = owner

    chunks = chunk_documents(docs)
    vs = VectorStore()

    if not is_repo:
        # Delete-then-insert, scoped to this owner only. Upsert alone never
        # removes anything, so a file deleted upstream — or a chunk index
        # that shrank because a file got shorter — would linger and keep
        # being retrieved. Scoping the delete to `owner` matters because two
        # different users can independently ingest the same URL/PDF; one
        # user's re-ingest must never delete another user's copy.
        removed = vs.delete_by_source(source, owner=owner)
        if removed:
            log.info("Removed %d stale chunks for %s before re-ingest", removed, source)
    # else: repo ingest is intentionally non-destructive — chunk ids are
    # tagged with commit_sha, so a new commit's chunks are ADDED alongside
    # chunks from older commits rather than overwriting them. Content dropped
    # in the latest commit stays retrievable. Re-ingesting the SAME commit is
    # still idempotent: the chunk ids are identical, so upsert just overwrites
    # those exact rows in place.

    added = vs.upsert(chunks)
    log.info("Ingested %d chunks in %.1fs — %d total in DB", added, time.time() - start, vs.count())
    return {"chunks_added": added, "kind": kind, "commit_sha": commit_sha}


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="GitHub URL, web URL, or PDF path")
    parser.add_argument("--owner", default=None, help="Tag ingested chunks as owned by this username")
    args = parser.parse_args()
    ingest(args.source, owner=args.owner)
