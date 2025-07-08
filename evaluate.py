import torch
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    pipeline,
    logging
)
from datasets import load_dataset
from rouge_score import rouge_scorer
import bert_score
import warnings

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

print("Load model and tokenizer")
print(f"CUDA: { torch.cuda.is_available()}")
model = MT5ForConditionalGeneration.from_pretrained("best_model")
tokenizer = AutoTokenizer.from_pretrained("best_model", use_fast=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Load test data")
test_dataset = load_dataset('json', data_files={'test': 'data/test.jsonl'})['test']

print("Summarization pipeline with headline task")
summarizer = pipeline(
    task="summarization",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    framework="pt",
    truncation=True
)

inputs = ["summarize: " + example["text"] for example in test_dataset]
references = [example["summary"] for example in test_dataset]

print("Optimized batch prediction")
predictions = []
batch_size = 4  # Adjust based on GPU memory

for i in range(0, len(inputs), batch_size):
    batch_inputs = inputs[i:i + batch_size]
    
    # Tokenize with padding and truncation
    tokenized = tokenizer(
        batch_inputs,
        padding=True,
        truncation=True,
        max_length=1024,  # Adjust based on model's max length
        return_tensors="pt"
    ).to(device)
    
    # Generate summaries
    summary_ids = model.generate(
        input_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
        max_length=100,  # Increased from 70 to 100
        min_length=30,    # Optional: Increase min length
        no_repeat_ngram_size=2,
        early_stopping=True,
        length_penalty=1.5,
        num_beams=4,  # Beam search often better than greedy
        do_sample=False
    )
    
    # Decode without special tokens + cleanup
    batch_preds = tokenizer.batch_decode(
        summary_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    predictions.extend(batch_preds)

# Calculate metrics
print("Calculate metrics")
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
rouge_scores = {
    "rouge1": {"precision": [], "recall": [], "f1": []},
    "rougeL": {"precision": [], "recall": [], "f1": []}
}

for ref, pred in zip(references, predictions):
    scores = scorer.score(ref, pred)
    # Extract precision, recall, f1 for each metric
    for metric in ["rouge1", "rougeL"]:
        rouge_scores[metric]["precision"].append(scores[metric].precision)
        rouge_scores[metric]["recall"].append(scores[metric].recall)
        rouge_scores[metric]["f1"].append(scores[metric].fmeasure)

_, _, bert_f1 = bert_score.score(
    predictions,
    references,
    lang="bg",
    model_type="bert-base-multilingual-cased",
    device=device
)

print("\n=== Evaluation Results ===")
# Print ROUGE-1 metrics
print(f"ROUGE-1 Precision: {sum(rouge_scores['rouge1']['precision']) / len(rouge_scores['rouge1']['precision']):.4f}")
print(f"ROUGE-1 Recall: {sum(rouge_scores['rouge1']['recall']) / len(rouge_scores['rouge1']['recall']):.4f}")
print(f"ROUGE-1 F1: {sum(rouge_scores['rouge1']['f1']) / len(rouge_scores['rouge1']['f1']):.4f}")

# Print ROUGE-L metrics
print(f"ROUGE-L Precision: {sum(rouge_scores['rougeL']['precision']) / len(rouge_scores['rougeL']['precision']):.4f}")
print(f"ROUGE-L Recall: {sum(rouge_scores['rougeL']['recall']) / len(rouge_scores['rougeL']['recall']):.4f}")
print(f"ROUGE-L F1: {sum(rouge_scores['rougeL']['f1']) / len(rouge_scores['rougeL']['f1']):.4f}")

print(f"BERTScore F1: {bert_f1.mean().item():.4f}")

# Save predictions
with open("predictions_samples.txt", "w", encoding="utf-8") as f:
    for i in range(len(test_dataset)):
        f.write(f"=== Example {i+1} ===\n")
        f.write(f"Original: {test_dataset[i]['text'][:500]}...\n")
        f.write(f"Reference: {test_dataset[i]['summary']}\n")
        f.write(f"Predicted: {predictions[i]}\n\n")