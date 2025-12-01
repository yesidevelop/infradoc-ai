import chromadb
from typing import List
from pathlib import Path
from sentence_transformers import SentenceTransformer
import fitz

DATA_DIR = "data"
CHROMA_DIR = "embeddings"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=CHROMA_DIR)
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

embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)
def create_embeddings (docs: List[dict]):
    for doc in docs:
        chunks = chunk_text(doc['content'])
        for i, chunk in enumerate(chunks):
            emb = embedding_model.encode(chunk)
            collection.add(
                documents = [chunk],
                metadatas = [{"source": doc["name"], "chunk": i}],
                ids = [f"{doc['name']}_{i}"]
            )
    
if __name__ == "__main__":
    docs = load_pdfs("data")
    create_embeddings(docs)
    
