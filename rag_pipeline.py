import chromadb
from typing import List
from pathlib import Path
from sentence_transformers import SentenceTransformer
import fitz

DATA_DIR = "data"
CHROMA_DIR = "embeddings"
# EMBEDDINGS_MODEL = "all-MiniLM-L6-v2" 
EMBEDDINGS_MODEL = "all-mpnet-base-v2" #Stronger one
embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)
# client = chromadb.Client()
client = chromadb.PersistentClient(path=CHROMA_DIR)
# client.delete_collection(name = "infra_docs")
collection = client.get_or_create_collection(name="infra_docs")

def load_pdfs(folder: str) -> List[str]:
    docs = []
    for file in Path(folder).glob("*.pdf"):
        pdf = fitz.open(file)
        text = ""
        for page in pdf:
            text += page.get_text()
        docs.append({"name": file.name, "content": text})
    return docs

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def create_embeddings (docs: List[dict]):
    existing_ids = set(collection.get()["ids"])
    for doc in docs:
        chunks = chunk_text(doc['content'])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['name']}_{i}"
            if chunk_id in existing_ids:
                print(f"Skipping duplicate chunks: {chunk_id}")
                continue
            emb = embedding_model.encode(chunk)
            collection.add(
                documents = [chunk],
                embeddings = [emb],
                metadatas = [{"source": doc["name"], "chunk": i}],
                ids = [f"{doc['name']}_{i}"]
            )
    
def semantic_search(query: str, top_k: int = 5):
    query_emb = embedding_model.encode(query)
    results = collection.query(query_embeddings = [query_emb], n_results = top_k)
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"Source: {meta['source']} | Chunk: {meta['chunk']}\n{doc}\n---")
if __name__ == "__main__":
    docs = load_pdfs("data")
    create_embeddings(docs)
    
    query = "Which is the database designed for handling time series?"
    # query = "Postgres Role change"
    semantic_search(query)