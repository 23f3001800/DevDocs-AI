from app.hybrid_retriever import HybridRetriever
from app.vectorstore import VectorStore

query = "How do I use the DevDocs API to get a list of all available documentation sets?"

# Dense only
vs = VectorStore()
dense_results = vs.query(query, k=1)

# Hybrid
hybrid = HybridRetriever()
hybrid_results = hybrid.retrieve(query, k=1)

print("\n=== DENSE ===")
print(dense_results[0]["content"][:1000])

print("\n=== HYBRID ===")
print(hybrid_results[0]["content"][:1000])
