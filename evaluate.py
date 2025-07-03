# evaluate.py
# requirements:
#   pip install rouge-score pandas

import json
from rouge_score import rouge_scorer
import pandas as pd

def load_results(path="results.json"):
    with open(path, encoding="utf‑8") as f:
        return json.load(f)

def evaluate(results):
    # Initialize a stemmer‑aware ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rows = []
    for item in results:
        ref = item['text']
        hyp = item['bg_summary']
        scores = scorer.score(ref, hyp)
        rows.append({
            'site':       item['site'],
            'url':        item['url'],
            'rouge-1':    scores['rouge1'].fmeasure,
            'rouge-L':    scores['rougeL'].fmeasure
        })
    df = pd.DataFrame(rows)
    # Compute corpus‑level average
    avg = df[['rouge-1','rouge-L']].mean().to_frame().T
    avg.insert(0, 'site', 'AVERAGE')
    avg.insert(1, 'url', '')
    return pd.concat([df, avg], ignore_index=True)

if __name__ == "__main__":
    results = load_results()
    df = evaluate(results)
    print(df.to_string(index=False))
    # Also save to CSV
    df.to_csv("rouge_evaluation.csv", index=False, float_format="%.4f")
    print("\nSaved detailed scores to rouge_evaluation.csv")
