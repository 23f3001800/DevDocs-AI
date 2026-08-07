import logging
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import git
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app.config import get_settings

log = logging.getLogger(__name__)

# File types we care about in a repo
CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs"}
DOC_EXTENSIONS = {".md", ".mdx", ".txt", ".rst"}
SKIP_DIRS = {"node_modules", ".git", "__pycache__", "dist", "build", ".venv", "venv"}


def load_github_repo(repo_url: str) -> list[Document]:
    """Clone a GitHub repo and load all code + doc files."""
    tmpdir = tempfile.mkdtemp()
    try:
        log.info("Cloning %s", repo_url)
        git.Repo.clone_from(repo_url, tmpdir, depth=1)
        docs = []
        for root, dirs, files in os.walk(tmpdir):
            # Slice assignment mutates the list os.walk reads, pruning the
            # traversal. `dirs = [...]` would rebind a local and change nothing.
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for file in files:
                ext = Path(file).suffix.lower()
                if ext not in CODE_EXTENSIONS | DOC_EXTENSIONS:
                    continue
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, tmpdir)
                try:
                    with open(filepath, encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if len(content.strip()) < 50:  # skip near-empty files
                        continue
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": repo_url,
                                "file_path": rel_path,
                                "file_type": ext.lstrip("."),
                                "is_code": ext in CODE_EXTENSIONS,
                            },
                        )
                    )
                except Exception:
                    continue
        log.info("Loaded %d files from %s", len(docs), repo_url)
        return docs
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def load_url(url: str) -> list[Document]:
    """Fetch a documentation page and extract its text content.

    WHY allow_redirects=False? The caller validates that `url` resolves to a
    public address, but a 302 to http://169.254.169.254/ would sail straight
    past that check — the redirect is followed by requests, after validation
    has already passed. Refusing to follow redirects closes that hole; the
    operator can pass the final URL directly.
    """
    settings = get_settings()
    try:
        r = requests.get(
            url,
            timeout=settings.ingest_timeout_seconds,
            headers={"User-Agent": "DevDocsAI/1.0"},
            allow_redirects=False,
        )
        if r.is_redirect or r.is_permanent_redirect:
            log.warning(
                "Refusing to follow redirect from %s to %r — pass the final URL directly.",
                url,
                r.headers.get("Location"),
            )
            return []
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        # Strip boilerplate: repeated nav/footer text appears in every chunk and
        # destroys BM25's IDF signal (a term in every doc carries no information).
        for tag in soup(["nav", "footer", "script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if not text.strip():
            log.warning("No text extracted from %s", url)
            return []

        # file_path must be set and unique per source — the vector store derives
        # chunk IDs from it, and a missing value used to collapse every URL onto
        # the same "doc_0, doc_1, ..." ID space (silently overwriting each other).
        parsed = urlparse(url)
        file_path = f"{parsed.netloc}{parsed.path}" or parsed.netloc
        return [
            Document(
                page_content=text,
                metadata={
                    "source": url,
                    "file_path": file_path.rstrip("/") or parsed.netloc,
                    "file_type": "html",
                    "is_code": False,
                },
            )
        ]
    except Exception as e:
        log.error("Failed to load %s: %s", url, e)
        return []


def load_pdf(path: str) -> list[Document]:
    """Load a local PDF file, one Document per page."""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(path)
    docs = loader.load()
    name = Path(path).name
    for i, d in enumerate(docs):
        d.metadata["is_code"] = False
        d.metadata["file_type"] = "pdf"
        d.metadata.setdefault("source", path)
        # PyPDFLoader sets `source` and `page` but never `file_path`; without it
        # every PDF would share the same generated chunk IDs.
        d.metadata["file_path"] = f"{name}#page={d.metadata.get('page', i)}"
    log.info("Loaded %d pages from %s", len(docs), path)
    return docs
