import os
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    def __init__(self, collection_name: str = "devdocs"):
        self.embedder = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        self.client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PATH", "./chroma_db")
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, docs: list[Document]) -> int:
        """Embed and store documents. Returns count added."""
        texts = [d.page_content for d in docs]
        embeddings = self.embedder.encode(
            texts, show_progress_bar=True, batch_size=64
        ).tolist()
        ids = [
            f"{d.metadata.get('file_path','doc')}_{d.metadata.get('chunk_index',i)}"
            for i, d in enumerate(docs)
        ]
        # Sanitise metadata — ChromaDB only accepts str/int/float/bool
        metadatas = []
        for d in docs:
            clean = {k: (str(v) if not isinstance(v,(str,int,float,bool))
                         else v)
                     for k,v in d.metadata.items()}
            metadatas.append(clean)

        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        return len(docs)
    
    def query(self, text: str, k: int = 5,
             file_type: str = None) -> list[dict]:
        """Query by semantic similarity, optional file_type filter."""
        q_embed = self.embedder.encode([text]).tolist()
        where = {"file_type": file_type} if file_type else None
        results = self.collection.query(
            query_embeddings=q_embed,
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["documents"][0]))
        ]

    def count(self) -> int:
        return self.collection.count()