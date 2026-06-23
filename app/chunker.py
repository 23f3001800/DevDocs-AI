from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents with awareness of file type.
    Code files use language-specific splitters.
    Docs use standard recursive splitter.
    """
    chunked = []
    for doc in docs:
        ft = doc.metadata.get("file_type", "")
        chunks = _split(doc, ft)
        chunked.extend(chunks)
    print(f"Chunked {len(docs)} docs → {len(chunked)} chunks")
    return chunked


def _split(doc: Document, file_type: str) -> list[Document]:
    # Map file types to LangChain Language enum
    lang_map = {
        "py": Language.PYTHON,
        "ts": Language.TS,
        "tsx": Language.TS,
        "js": Language.JS,
        "jsx": Language.JS,
        "go": Language.GO,
        "rs": Language.RUST,
    }
    if file_type in lang_map:
        # Code-aware split: respects function/class boundaries
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_map[file_type],
            chunk_size=1200,  # larger for code — functions can be long
            chunk_overlap=100,
        )
    else:
        # Standard split for markdown, HTML, plain text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=60, separators=["\n\n", "\n", " ", ""]
        )
    chunks = splitter.split_documents([doc])
    # Preserve + add metadata to every chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    return chunks
