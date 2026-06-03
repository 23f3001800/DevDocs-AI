# DevDocs-AI
RAG over any GitHub repo or docs site — for developers


python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

python -c "from app.vectorstore import VectorStore; vs = VectorStore(); results = vs.query('how to create a route', k=3); print(results[0]['content'][:200])"