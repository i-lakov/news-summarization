import json
from rouge_score import rouge_scorer
import bert_score
import re

def lead_1(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return sentences[0] if sentences else ""

def title_baseline(title: str):
    return title.strip()

with open("data/test.jsonl", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

references = [d["summary"] for d in data]
lead1_preds = [lead_1(d["text"]) for d in data]
title_preds = [title_baseline(d["title"]) for d in data]

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

def compute_rouge_all(preds):
    scores = {
        "rouge1": {"precision": [], "recall": [], "fmeasure": []},
        "rougeL": {"precision": [], "recall": [], "fmeasure": []}
    }
    for ref, pred in zip(references, preds):
        result = scorer.score(ref, pred)
        for key in ["rouge1", "rougeL"]:
            scores[key]["precision"].append(result[key].precision)
            scores[key]["recall"].append(result[key].recall)
            scores[key]["fmeasure"].append(result[key].fmeasure)
    return {
        key: {
            "precision": sum(val["precision"]) / len(val["precision"]),
            "recall": sum(val["recall"]) / len(val["recall"]),
            "f1": sum(val["fmeasure"]) / len(val["fmeasure"])
        }
        for key, val in scores.items()
    }

# ROUGE scores
rouge_lead1 = compute_rouge_all(lead1_preds)
rouge_title = compute_rouge_all(title_preds)

# BERTScore
_, _, bert_lead1 = bert_score.score(lead1_preds, references, lang="bg", model_type="bert-base-multilingual-cased", verbose=True)
_, _, bert_title = bert_score.score(title_preds, references, lang="bg", model_type="bert-base-multilingual-cased", verbose=True)

def print_metrics(name, rouge, bert_f1):
    print(f"[{name}]")
    print(f"ROUGE-1 Precision: {rouge['rouge1']['precision']:.4f}")
    print(f"ROUGE-1 Recall:    {rouge['rouge1']['recall']:.4f}")
    print(f"ROUGE-1 F1:        {rouge['rouge1']['f1']:.4f}")
    print(f"ROUGE-L Precision: {rouge['rougeL']['precision']:.4f}")
    print(f"ROUGE-L Recall:    {rouge['rougeL']['recall']:.4f}")
    print(f"ROUGE-L F1:        {rouge['rougeL']['f1']:.4f}")
    print(f"BERTScore F1:      {bert_f1.mean().item():.4f}")
    print()

print("=== Baseline Evaluation ===")
print_metrics("Lead-1", rouge_lead1, bert_lead1)
print_metrics("Title", rouge_title, bert_title)

with open("baseline_predictions.txt", "w", encoding="utf-8") as f:
    for i, (d, lead, title) in enumerate(zip(data, lead1_preds, title_preds)):
        f.write(f"=== Example {i+1} ===\n")
        f.write(f"Original: {d['text'][:500]}...\n")
        f.write(f"Reference: {d['summary']}\n")
        f.write(f"Lead-1: {lead}\n\n")
        f.write(f"Title: {title}\n\n")
