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

inputs = ["headline: " + example["text"] for example in test_dataset]
references = [example["summary"] for example in test_dataset]

print("Batch prediction")
predictions = []
batch_size = 8

for i in range(0, len(inputs), batch_size):
    batch = inputs[i:i + batch_size]
    results = summarizer(
        batch,
        max_length=70,
        min_length=5,
        do_sample=False,
        truncation=True,
        no_repeat_ngram_size=2,
        early_stopping=True,
        length_penalty=1.0
    )
    predictions.extend([result["summary_text"] for result in results])

# Calculate metrics
print("Calculate metrics")
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
rouge_scores = {"rouge1": [], "rougeL": []}

for ref, pred in zip(references, predictions):
    scores = scorer.score(ref, pred)
    rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
    rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

_, _, bert_f1 = bert_score.score(
    predictions,
    references,
    lang="bg",
    model_type="bert-base-multilingual-cased",
    device=device
)

print("\n=== Evaluation Results ===")
print(f"ROUGE-1: {sum(rouge_scores['rouge1'])/len(rouge_scores['rouge1']):.4f}")
print(f"ROUGE-L: {sum(rouge_scores['rougeL'])/len(rouge_scores['rougeL']):.4f}")
print(f"BERTScore F1: {bert_f1.mean().item():.4f}")

# Save predictions
with open("predictions_samples.txt", "w", encoding="utf-8") as f:
    for i in range(len(test_dataset)):
        f.write(f"=== Example {i+1} ===\n")
        f.write(f"Original: {test_dataset[i]['text'][:500]}...\n")
        f.write(f"Reference: {test_dataset[i]['summary']}\n")
        f.write(f"Predicted: {predictions[i]}\n\n")