"""Index only the PDF's 13 source documents, never its questions or answer key."""
import json
import re
from pathlib import Path


def extract_sources(pages):
    text = "\n".join(pages)
    start = text.index("3. Mini Technical Corpus")
    end = re.search(r"4\.\s+", text[start:])
    if end is None:
        raise ValueError("Missing corpus boundary")
    text = text[start:start + end.start()]
    text = re.sub(r"(?m)^Page \d+\s*$", "", text).replace("•", "")
    headers = list(re.finditer(r"(?m)^(doc_[a-z]+_\d+)\s+(docs/\S+)\s*$", text))
    if len(headers) != 13:
        raise ValueError(f"Expected 13 source documents, found {len(headers)}")
    docs = []
    for i, header in enumerate(headers):
        body = text[header.end():headers[i + 1].start() if i + 1 < len(headers) else len(text)].strip()
        if any(marker in body for marker in ("TC-01", "Expected docs:", "Ground-truth answer:")):
            raise ValueError("Answer-key contamination")
        docs.append({"doc_id": header[1], "source_path": header[2], "content": body})
    return docs


def main():
    from langchain_core.documents import Document
    from pypdf import PdfReader

    from app.chunker import chunk_documents
    from app.vectorstore import VectorStore

    sources = extract_sources([p.extract_text() for p in PdfReader("data/devdocs_ragas_eval_test_cases.pdf").pages])
    cases = json.loads(Path("data/eval_dataset.json").read_text())
    ids = {source["doc_id"] for source in sources}
    assert all(set(c["ground_truth_doc_ids"]) <= ids for c in cases)
    documents = [Document(page_content=s["content"], metadata={
        "doc_id": s["doc_id"], "file_path": s["source_path"],
        "source": "evaluation-source-corpus", "file_type": "md",
    }) for s in sources]
    vs = VectorStore()
    assert vs.count() == 0, "Benchmark requires a clean index"
    count = vs.upsert(chunk_documents(documents))
    out = Path("benchmark-results")
    out.mkdir(exist_ok=True)
    (out / "source-corpus.json").write_text(json.dumps(sources, indent=2))
    print(f"Indexed {len(sources)} source documents as {count} chunks; answer key excluded")


if __name__ == "__main__":
    main()
