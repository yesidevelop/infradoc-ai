import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

DATA_PATH = "training_data/merged/final_dataset.jsonl"
LORA_PATH = "fine_tuned_model"
OUTPUT_DIR = "merged-model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODEL = "NousResearch/Hermes-3-Llama-3.2-3B"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map = "auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = PeftModel.from_pretrained(base_model, LORA_PATH)

model = model.merge_and_unload()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
