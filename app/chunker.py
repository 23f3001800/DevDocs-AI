import logging

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.config import get_settings

log = logging.getLogger(__name__)

# Map file types to LangChain's Language enum. Language-specific splitters use
# separators like "\nclass " and "\ndef " *before* the generic ones, so chunks
# break at function/class boundaries and stay self-contained.
LANG_MAP = {
    "py": Language.PYTHON,
    "ts": Language.TS,
    "tsx": Language.TS,
    "js": Language.JS,
    "jsx": Language.JS,
    "go": Language.GO,
    "rs": Language.RUST,
}


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents with awareness of file type.

    Chunk size is the highest-leverage RAG hyperparameter: too small and a fact
    is split across chunks that never co-retrieve; too large and one relevant
    sentence is diluted by noise. Sizes are configurable (CHUNK_SIZE_CODE /
    CHUNK_SIZE_TEXT) so they can be tuned.
    """
    chunked = []
    for doc in docs:
        chunked.extend(_split(doc, doc.metadata.get("file_type", "")))
    log.info("Chunked %d docs → %d chunks", len(docs), len(chunked))
    return chunked


def _split(doc: Document, file_type: str) -> list[Document]:
    settings = get_settings()
    if file_type in LANG_MAP:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=LANG_MAP[file_type],
            chunk_size=settings.chunk_size_code,  # larger — functions can be long
            chunk_overlap=settings.chunk_overlap_code,
        )
    else:
        # Standard split for markdown, HTML, plain text. Separators are tried in
        # priority order, so it breaks at the most semantically meaningful
        # boundary available.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size_text,
            chunk_overlap=settings.chunk_overlap_text,
            separators=["\n\n", "\n", " ", ""],
        )
    chunks = splitter.split_documents([doc])
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    return chunks
