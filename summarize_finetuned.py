import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer, pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "best_model"
model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

summarizer = pipeline(
    task="summarization",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    framework="pt"
)

def summarize_bg(text: str, min_length: int = 15, max_length: int = 60) -> dict:
    if not text.strip() or len(text.strip()) < 50:
        return {"bg_summary": ""}

    try:
        inputs = "headline: " + text
        summary = summarizer(
            inputs,
            max_length=max_length,
            min_length=min_length,
            truncation=True,
            no_repeat_ngram_size=2,
            early_stopping=True,
            length_penalty=1.0,
            num_beams=4
        )[0]["summary_text"]

        return {"bg_summary": summary}
    except Exception as e:
        print(f"Summarization failed: {e}")
        return {"bg_summary": ""}
