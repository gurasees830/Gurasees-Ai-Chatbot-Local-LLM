"""LoRA fine-tune a coding model on train.jsonl, then export a GGUF for Ollama.

Requires an NVIDIA GPU (>= 8 GB VRAM for a 3B model, >= 16 GB for 7B), or a free
Google Colab T4 GPU. Install:   pip install unsloth

Run:  python finetune.py
Output: ./gguf_out/  (contains a .gguf file you import into Ollama, see README)
"""
from unsloth import FastLanguageModel  # import first: it patches transformers/trl

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# Pick a base that matches what you run in Ollama. Smaller GPU? use the 3B variant.
BASE_MODEL = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"  # 3B: small enough to run on a 4 GB GPU afterwards
MAX_SEQ_LEN = 4096
DATA_FILE = "train.jsonl"
EPOCHS = 3
LEARNING_RATE = 2e-4

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)

# Attach small trainable LoRA adapters instead of updating all weights.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.map(
    lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)},
    remove_columns=dataset.column_names,
)
print(f"Training on {len(dataset)} examples")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=5,
        optim="adamw_8bit",
        output_dir="outputs",
        report_to="none",
        seed=42,
    ),
)
trainer.train()

# Merge the adapters and export a 4-bit GGUF that Ollama can load.
model.save_pretrained_gguf("gguf_out", tokenizer, quantization_method="q4_k_m")
print("Done. Look inside ./gguf_out (or ./gguf_out_gguf) for the .gguf file.")
