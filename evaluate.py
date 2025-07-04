# evaluate.py
# requirements:
#   pip install rouge-score pandas bert-score

import json
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score

class SplitTokenizer:
    """Simple whitespace tokenizer for RougeScorer."""
    def tokenize(self, text: str):
        return text.split()

def load_results(path="results.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def evaluate_with_rouge(results):
    """
    Returns a list of dicts:
      { 'rouge-1': float, 'rouge-L': float }
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        use_stemmer=False,
        tokenizer=SplitTokenizer()
    )
    rows = []
    for item in results:
        ref, hyp = item["text"], item["bg_summary"]
        scores = scorer.score(ref, hyp)
        rows.append({
            "rouge-1": scores["rouge1"].fmeasure,
            "rouge-L": scores["rougeL"].fmeasure
        })
    return rows

def evaluate_with_bertscore(results):
    """
    Returns a list of BERTScore F1 floats.
    """
    refs = [it["text"] for it in results]
    hyps = [it["bg_summary"] for it in results]
    # lang="bg" uses a multilingual checkpoint under the hood
    P, R, F1 = bert_score(hyps, refs, lang="bg", model_type="xlm-roberta-base", verbose=False)
    return F1.tolist()

if __name__ == "__main__":
    results = load_results()

    # Compute metrics
    rouge_rows = evaluate_with_rouge(results)
    bert_f1s   = evaluate_with_bertscore(results)

    # Build DataFrame
    records = []
    for item, rr, bf in zip(results, rouge_rows, bert_f1s):
        records.append({
            "site":     item["site"],
            "url":      item["url"],
            "rouge-1":  rr["rouge-1"],
            "rouge-L":  rr["rouge-L"],
            "bert-F1":  bf
        })
    df = pd.DataFrame(records)

    # Compute average row
    avg = df[["rouge-1", "rouge-L", "bert-F1"]].mean().to_frame().T
    avg.insert(0, "url", "")
    avg.insert(0, "site", "AVERAGE")

    # Final table
    final = pd.concat([df, avg], ignore_index=True)

    # Display
    print(final.to_string(index=False, float_format="%.4f"))

    # Save
    final.to_csv("evaluation.csv", index=False, float_format="%.4f")
    print("\nSaved detailed scores to evaluation.csv")
