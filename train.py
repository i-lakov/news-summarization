import json
import torch
import transformers
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt  # Added for plotting

# Verify environments
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NumPy version: {np.__version__}")

# Load dataset
dataset = load_dataset('json', data_files={
    'train': 'data/train.jsonl',
    'validation': 'data/validation.jsonl'
})

# Initialize mT5 model
model = MT5ForConditionalGeneration.from_pretrained(
    "google/mt5-base",
    use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "google/mt5-base",
    use_fast=True
)

# Preprocessing function with task prefix and filtering
def preprocess_function(examples):
    inputs = []
    targets = []
    for i in range(len(examples["text"])):
        text = examples["text"][i].strip()
        summary = examples["summary"][i].strip()
        if not text or not summary or len(summary.split()) < 3:
            continue
        inputs.append("headline: " + text)
        targets.append(summary)
    
    model_inputs = tokenizer(
        inputs,
        max_length=1024,
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=25,
            truncation=True,
            padding="max_length"
        )
    labels = labels["input_ids"]
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in seq]
        for seq in labels
    ]
    model_inputs["labels"] = labels
    return model_inputs

# Tokenize dataset
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=4,
    remove_columns=dataset["train"].column_names
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    max_length=1024,
    pad_to_multiple_of=8
)

# Training arguments (Improved)
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=8,
    max_grad_norm=1.0,
    logging_steps=10,
    report_to="none",
    optim="adafactor",
    warmup_ratio=0.1,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# Loss callback with tracking
class LossCallback(transformers.TrainerCallback):
    def __init__(self):
        self.losses = []
        self.steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            loss = logs["loss"]
            step = state.global_step
            self.losses.append(loss)
            self.steps.append(step)
            if torch.isnan(torch.tensor(loss)):
                print(f"NaN loss detected at step {state.global_step}! Stopping training.")
                control.should_training_stop = True

loss_callback = LossCallback()
trainer.add_callback(loss_callback)

# Train
trainer.train()

# Save model & tokenizer
trainer.save_model("best_model")
tokenizer.save_pretrained("best_model")

# Plot training curve
plt.figure(figsize=(10, 6))
plt.plot(loss_callback.steps, loss_callback.losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("training_curve.png")  # Saves to file
plt.show()