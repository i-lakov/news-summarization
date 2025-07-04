import logging
from transformers import (
    MarianTokenizer, MarianMTModel,
    pipeline as hf_pipeline,
    AutoTokenizer
)
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1) BG -> EN translation
BG_EN_MODEL = "Helsinki-NLP/opus-mt-bg-en"
tok_bg_en   = MarianTokenizer.from_pretrained(BG_EN_MODEL, use_fast=False)
model_bg_en = MarianMTModel.from_pretrained(BG_EN_MODEL).to(DEVICE)

# 2) English summarizer
SUM_MODEL   = "facebook/bart-large-cnn"
summarizer  = hf_pipeline(
    "summarization",
    model=SUM_MODEL,
    device=0 if DEVICE=="cuda" else -1
)

# 3) EN -> BG translation
EN_BG_MODEL = "Helsinki-NLP/opus-mt-en-bg"
tok_en_bg   = MarianTokenizer.from_pretrained(EN_BG_MODEL, use_fast=False)
model_en_bg = MarianMTModel.from_pretrained(EN_BG_MODEL).to(DEVICE)

summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def translate(text: str, tokenizer, model, max_length: int = 512) -> str:
    hard_max = tokenizer.model_max_length
    use_len = min(max_length, hard_max)
 
    # helper to do one pass of tokenize + generate
    def _gen_input(text, length):
        batch = tokenizer.prepare_seq2seq_batch(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=length
        ).to(DEVICE)
        out = model.generate(
            **batch,
            max_length=length,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.batch_decode(out, skip_special_tokens=True)[0]
 
    try:
        return _gen_input(text, use_len)
    except IndexError:
        # retry with a shorter length
        shorter = max(use_len // 2, 32)
        return _gen_input(text, shorter)

def summarize_bg(text: str, en_min_length=40, en_max_length=200, bg_max_length=150) -> dict:
    if len(text) < 100:
        logging.warning("Skipping too short article.")
        return {"en_summary": "", "bg_summary": ""}

    if not text.strip():
        return {"en_summary": "", "bg_summary": ""}

    try:
        # Step 1: Translate BG -> EN
        en = translate(text, tok_bg_en, model_bg_en, max_length=600)

        # Step 2: Tokenize EN translation
        input_tokens = summarizer_tokenizer.encode(en, truncation=True, max_length=1024)
        input_len = len(input_tokens)

        # Summary should be ~30–60% of the input length
        max_len = min(int(input_len * 0.6), en_max_length)
        min_len = min(int(input_len * 0.3), en_min_length)

        # Step 3: Summarize EN
        summary = summarizer(
            en,
            max_length=max_len,
            min_length=min_len,
            truncation=True
        )[0]['summary_text']

        # Step 4: Translate EN summary -> BG
        bg = translate(summary, tok_en_bg, model_en_bg, max_length=bg_max_length)

        return {
            "en_summary": summary,
            "bg_summary": bg
        }

    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        return {"en_summary": "", "bg_summary": ""}

if __name__ == "__main__":
    sample = (
        "България влиза в ЕС през 2007 година. "
        "Страната е член на НАТО от 2004 година. "
        "Икономиката ѝ се развива бързо в последните години."
    )
    out = summarize_bg(sample, en_min_length=50, en_max_length=150, bg_max_length=180)
    print("EN:", out["en_summary"])
    print("BG:", out["bg_summary"])
