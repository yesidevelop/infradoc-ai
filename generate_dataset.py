import os
from sentence_transformers import SentenceTransformer
import fitz
import json

DOCS_PATH = "./data"
OUTPUT_PATH = "./training_data"

os.makedirs(OUTPUT_PATH, exist_ok=True)

os.makedirs(f"{OUTPUT_PATH}/sft", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/instruction", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/rag", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/merged", exist_ok=True)

EMBEDDINGS_MODEL = "all-mpnet-base-v2" #Stronger one
embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)

CHUNK_SIZE = 500

def read_pdf(file_path: str):
    
    pdf = fitz.open(file_path)
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

def chunk_text(text: str, size: int = 500):
    chunks = []
    for i in range(0, len(text), size):
        chunks.append( text[i:i + size])
    return chunks


sft_samples = []
instruction_samples = []
rag_samples = []


for file_name in os.listdir(DOCS_PATH):
    file_path = os.path.join(DOCS_PATH, file_name)
    if not os.path.isfile(file_path):
        continue

    text = read_pdf(file_path)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        sft_samples.append({
            "instruction": f"Explain this snippet from {file_name}",
            "input": "",
            "output": chunk
        })

        instruction_samples.append({
            "instruction": f"Summarize this text in Bullet points",
            "input": chunk,
            "output": "- " + "\n- " .join(chunk.split(". ")[:5])
        })

        rag_samples.append({
            "instruction": f"Answer the question using only the context",
            "input": f"CONTEXT: {chunk}",
            "output": f"ANSWER: {chunk[:min(200, len(chunk))]}..."
        })


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


save_jsonl(sft_samples, f"{OUTPUT_PATH}/sft/domain_sft.jsonl")
save_jsonl(instruction_samples, f"{OUTPUT_PATH}/instruction/instruction_data.jsonl")
save_jsonl(rag_samples, f"{OUTPUT_PATH}/rag/rag_qa.jsonl")

merged = sft_samples + instruction_samples + rag_samples

save_jsonl(merged, f"{OUTPUT_PATH}/merged/final_dataset.jsonl")
print(f"Dataset generation complete. Total samples: {len(merged)}")