import logging
import time
from scraping import scrape_all, dedupe_articles
from summarize import summarize_bg
import json

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def summarize_news(max_per_site=5,
                   dedupe_threshold=0.85,
                   en_min_length=40,
                   en_max_length=200,
                   bg_max_length=200):
    """
    1. Scrape articles from all configured sites (max_per_site each)
    2. Deduplicate by cosine similarity
    3. Summarize each unique article
    Returns list of dicts: {
      'site','url','title','en_summary','bg_summary'
    }
    """
    logging.info(f"Scraping up to {max_per_site} articles per site…")
    raw_articles = scrape_all(max_per_site)
    logging.info(f"Fetched {len(raw_articles)} total articles.")

    logging.info("Deduplicating articles…")
    unique_articles = dedupe_articles(raw_articles, threshold=dedupe_threshold)
    logging.info(f"{len(unique_articles)} articles remain after deduplication.")

    results = []
    for idx, art in enumerate(unique_articles, start=1):
        logging.info(f"Summarizing [{idx}/{len(unique_articles)}]: {art['title'][:60]}…")
        start = time.time()
        try:
            out = summarize_bg(
                art['text'],
                en_min_length=en_min_length,
                en_max_length=en_max_length,
                bg_max_length=bg_max_length
            )
            elapsed = time.time() - start
            if out["bg_summary"]:
                results.append({
                    'site': art['site'],
                    'url': art['url'],
                    'title': art['title'],
                    'text': art['text'], 
                    'en_summary': out['en_summary'],
                    'bg_summary': out['bg_summary']
                })
                logging.info(f"✓ Done in {elapsed:.1f}s.")
            else:
                logging.warning(f"Empty summary for URL: {art['url']}")
        except Exception as e:
            logging.error(f"Error summarizing {art['url']}: {e}")

    return results

if __name__ == '__main__':
    configure_logging()
    logging.info("Starting news summarization pipeline…")

    MAX_PER_SITE     = 3
    DEDUPE_THRESHOLD = 0.85
    EN_MIN_LEN       = 80    # tokens
    EN_MAX_LEN       = 300   # tokens
    BG_MAX_LEN       = 300   # tokens

    summaries = summarize_news(
        MAX_PER_SITE,
        DEDUPE_THRESHOLD,
        en_min_length=EN_MIN_LEN,
        en_max_length=EN_MAX_LEN,
        bg_max_length=BG_MAX_LEN
    )

    with open("results.json", "w", encoding="utf‑8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print("FINAL SUMMARIES:\n")
    for i, item in enumerate(summaries, start=1):
        print(f"#{i} [{item['site']}] {item['title']}")
        print(f"EN summary: {item['en_summary']}\n")
        print(f"BG summary: {item['bg_summary']}\n")
        print(f"URL: {item['url']}\n" + "-"*80)
