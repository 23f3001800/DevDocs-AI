import pypdf
import sys

reader = pypdf.PdfReader('data/devdocs_ragas_eval_test_cases.pdf')
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

print(f"Total text length: {len(text)}")
print("doc_embeddings_03 found:", "doc_embeddings_03" in text)
