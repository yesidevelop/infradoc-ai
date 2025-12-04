import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model

DATA_PATH = "training_data/merged/final_dataset.jsonl"
OUTPUT_DIR = "fine_tuned_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODEL = "NousResearch/Hermes-3-Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token


dataset = load_dataset("json", data_files = DATA_PATH, split = "train")

def tokize_fn(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output_text = example["output"]
    prompt = f"### Instruction:\n{instruction}\n### Input: \n {input_text}\n### Response\n{output_text}"
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokize_fn, remove_columns=dataset.column_names)



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map = "auto",
    quantization_config=bnb_config
)

lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout = 0.1,
    bias = "none",
    target_modules = ["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    num_train_epochs=1,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"fine tuned model saved ini {OUTPUT_DIR}")