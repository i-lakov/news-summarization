import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer from fine-tuned path
MODEL_PATH = "best_model"
model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

# Generation parameters (same as in evaluate.py)
GEN_CONFIG = {
    "max_length": 100,
    "min_length": 30,
    "no_repeat_ngram_size": 2,
    "early_stopping": True,
    "length_penalty": 1.5,
    "num_beams": 4,
    "do_sample": False
}

def summarize_bg(text: str, min_length: int = 30, max_length: int = 100) -> dict:
    if not text.strip() or len(text.strip()) < 50:
        return {"bg_summary": ""}

    try:
        # Apply task prefix
        prompt = "summarize: " + text

        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(DEVICE)

        # Override length params (if needed)
        gen_config = GEN_CONFIG.copy()
        gen_config["max_length"] = max_length
        gen_config["min_length"] = min_length

        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **gen_config
            )

        # Decode
        summary_text = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return {"bg_summary": summary_text}

    except Exception as e:
        print(f"[summarize_bg] Generation failed: {e}")
        return {"bg_summary": ""}
