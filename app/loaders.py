import os, tempfile, shutil
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader,
    UnstructuredMarkdownLoader, BSHTMLLoader
)
import git, requests
from bs4 import BeautifulSoup


# File types we care about in a repo
CODE_EXTENSIONS = {'.py','.ts','.tsx','.js','.jsx','.go','.rs'}
DOC_EXTENSIONS  = {'.md','.mdx','.txt','.rst'}
SKIP_DIRS       = {'node_modules','.git','__pycache__','dist','build','.venv','venv'}

def load_github_repo(repo_url: str) -> list[Document]:
    """Clone a GitHub repo and load all code + doc files."""
    tmpdir = tempfile.mkdtemp()
    try:
        print(f"Cloning {repo_url}...")
        git.Repo.clone_from(repo_url, tmpdir, depth=1)
        docs = []
        for root, dirs, files in os.walk(tmpdir):
            # Skip unwanted directories in-place
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for file in files:
                ext = Path(file).suffix.lower()
                if ext not in CODE_EXTENSIONS | DOC_EXTENSIONS:
                    continue
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, tmpdir)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if len(content.strip()) < 50:  # skip near-empty files
                        continue
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": repo_url,
                            "file_path": rel_path,
                            "file_type": ext.lstrip('.'),
                            "is_code": ext in CODE_EXTENSIONS
                        }
                    ))
                except Exception:
                    continue
        print(f"Loaded {len(docs)} files from {repo_url}")
        return docs
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def load_url(url: str, max_pages: int = 20) -> list[Document]:
    """Crawl a documentation URL and load text content."""
    try:
        r = requests.get(url, timeout=10,
            headers={"User-Agent": "DevDocsAI/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        # Remove nav, footer, scripts
        for tag in soup(['nav','footer','script','style']):
            tag.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return [Document(
            page_content=text,
            metadata={"source": url, "file_type": "html", "is_code": False}
        )]
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return []

def load_pdf(path: str) -> list[Document]:
    """Load a local PDF file."""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(path)
    docs = loader.load()
    for d in docs:
        d.metadata["is_code"] = False
        d.metadata["file_type"] = "pdf"
    return docs