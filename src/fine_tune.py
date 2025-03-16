import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load model & tokenizer
model_name = "deepseek-r1:8b"  # Replace with your locally hosted model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use Metal backend for Apple M3
device = torch.device("mps")  # Apple Metal Performance Shaders
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map={"": device}
)

# Load dataset (Extracted from pgvector or PDFs)
dataset = load_dataset("json", data_files="data/fine_tuning_data.jsonl")

# Apply LoRA for efficient tuning
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

# Training arguments optimized for M3 Pro
training_args = TrainingArguments(
    output_dir="models/fine_tuned_llm",
    per_device_train_batch_size=1,  # Adjusted for 16GB RAM
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=1e-4,
    num_train_epochs=3,
    fp16=True,  # Use 16-bit precision for better Apple Metal optimization
    optim="adamw_torch",
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()
model.save_pretrained("models/fine_tuned_llm")
print("✅ Fine-Tuning Complete!")
