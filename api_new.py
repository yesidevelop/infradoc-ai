import chromadb
from sentence_transformers import SentenceTransformer
import requests
from fastapi import FastAPI, Query as FastAPIQuery
from pydantic import BaseModel
from chromadb.config import Settings
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

embedding_model = SentenceTransformer("all-mpnet-base-v2")

# client = chromadb.Client(Settings(chroma_db_impl="douckdb+parquet", persist_directory="./chroma_db"))
CHROMA_DIR = "embeddings"
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="infra_docs")
BASE_MODEL = "NousResearch/Hermes-3-Llama-3.2-3B"
FINETUNED_DIR="./fine_tuned_model"

app = FastAPI()
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map = "auto",
    quantization_config=bnb_config
)

model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)
model.eval()




DATA_DIR = "data"
CHROMA_DIR = "embeddings"


class Query(BaseModel):
    question: str


@app.post("/query")
def rag_query(payload: Query):
    question = payload.question

    embedder = SentenceTransformer("all-mpnet-base-v2")
    q_emb = embedder.encode(question).tolist()
    
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=3
    )

    retrieved_docs = results["documents"][0]

    context_block = "\n\n".join(retrieved_docs)

    print(context_block)
    print(question)

    prompt = f"""
You are an expert assistant for internal documentaation and infra.

Use ONLY the context below to answer.

Context:
{context_block}

Question:
{question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 350,
            temperature = 0.2,
            top_p = 0.9
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens = True)
    answer = answer.split("Answer:")[-1].strip()

    return {"answer": answer, "context": retrieved_docs}