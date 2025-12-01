import chromadb
from sentence_transformers import SentenceTransformer
import requests
from fastapi import FastAPI, Query as FastAPIQuery
from pydantic import BaseModel

embedding_model = SentenceTransformer("all-mpnet-base-v2")

client = chromadb.Client()
collection_name = "infra_docs"

DATA_DIR = "data"
CHROMA_DIR = "embeddings"
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="infra_docs")

app = FastAPI()

class Query(BaseModel):
    question: str



def semantic_search(query: str, top_k: int = 3):
    query_emb = embedding_model.encode(query)
    results = collection.query(query_embeddings = [query_emb], n_results = top_k)
    return results['documents'][0]

def call_ollama(prompt: str):
    resp = requests.post(
        "http://localhost:11434/v1/completions",
        json = {
            "model": "llama3.2",
            "prompt": prompt,
            "max_tokens": 500
        }
    )
    data = resp.json()
    print(data)
    return data['choices'][0]['text']

@app.get("/")
def home():
    return {"text": "I am home"}

@app.post("/rag")
def rag_endpoint(query: Query):
    retrieved_docs = semantic_search(query.question, top_k = 3)

    context_text = "\n".join(retrieved_docs)
    prompt = f"Answer the following questions using the context below:\n\nContext:\n{context_text}\n\nQuestion: {query.question} \n Answer:"

    answer = call_ollama(prompt)
    return {"answer": answer}