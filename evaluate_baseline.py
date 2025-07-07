import json
from rouge_score import rouge_scorer
import bert_score
import re

def lead_1(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return sentences[0] if sentences else ""

def title_baseline(title: str):
    return title.strip()

# Load test set
with open("data/test.jsonl", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Evaluate both baselines
references = [d["summary"] for d in data]
lead1_preds = [lead_1(d["text"]) for d in data]
title_preds = [title_baseline(d["title"]) for d in data]

# ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

def compute_rouge(preds):
    rouge1, rougeL = [], []
    for ref, pred in zip(references, preds):
        scores = scorer.score(ref, pred)
        rouge1.append(scores["rouge1"].fmeasure)
        rougeL.append(scores["rougeL"].fmeasure)
    return sum(rouge1)/len(rouge1), sum(rougeL)/len(rougeL)

# BERTScore
_, _, bert_lead1 = bert_score.score(lead1_preds, references, lang="bg", model_type="bert-base-multilingual-cased", verbose=True)
_, _, bert_title = bert_score.score(title_preds, references, lang="bg", model_type="bert-base-multilingual-cased", verbose=True)

# Results
r1_l1, rL_l1 = compute_rouge(lead1_preds)
r1_title, rL_title = compute_rouge(title_preds)

print("=== Baseline Evaluation ===")
print(f"[Lead-1]    ROUGE-1: {r1_l1:.4f}  ROUGE-L: {rL_l1:.4f}  BERTScore F1: {bert_lead1.mean().item():.4f}")
print(f"[Title]     ROUGE-1: {r1_title:.4f}  ROUGE-L: {rL_title:.4f}  BERTScore F1: {bert_title.mean().item():.4f}")

with open("baseline_predictions.txt", "w", encoding="utf-8") as f:
    for i, (d, lead, title) in enumerate(zip(data, lead1_preds, title_preds)):
        f.write(f"=== Example {i+1} ===\n")
        f.write(f"Original: {d['text'][:500]}...\n")
        f.write(f"Reference: {d['summary']}\n")
        f.write(f"Lead-1: {lead}\n\n")
        f.write(f"Title: {title}\n\n")
