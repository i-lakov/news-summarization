import feedparser
from newspaper import Article
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer, util
import torch

SITES = {
    "bnt": {
        "rss": "https://news.bnt.bg/bg/rss/news.xml",
        "title_sel": "h1.article__title",
        "body_sel": "div.article__text p"
    },
    "svobodnaevropa": {
        "rss": "https://www.svobodnaevropa.bg/api/epiqq",
        "title_sel": "h1.title",
        "body_sel": "div.news__text p"
    },
    "standart": {
        "rss": "https://www.standartnews.com/rss?p=1",
        "title_sel": "h1.article-title",
        "body_sel": "div.article-body p"
    }
}

def get_article_urls(rss_url, max_articles):
    feed = feedparser.parse(rss_url)
    return [entry.link for entry in feed.entries[:max_articles]]

def fetch_full_text(url, cfg):
    # Try newspaper3k first
    art = Article(url, language='bg')
    art.download(); art.parse()
    if len(art.text) > 200:
        return art.title, art.text

    # Fallback to BS4 with site‐specific selectors - currently unused in reality
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.content, 'html.parser')
    title = soup.select_one(cfg["title_sel"]).get_text(strip=True)
    paras = soup.select(cfg["body_sel"])
    body = "\n\n".join(p.get_text(strip=True) for p in paras)
    return title, body

def scrape_all(max_per_site=10):
    all_articles = []
    for name, cfg in SITES.items():
        urls = get_article_urls(cfg["rss"], max_per_site)
        for u in urls:
            try:
                title, text = fetch_full_text(u, cfg)
                all_articles.append({
                    "site": name, "url": u, "title": title, "text": text
                })
            except Exception as e:
                print(f"[{name}] Error fetching {u}: {e}")
    return all_articles

def dedupe_articles(articles, threshold=0.85, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    # Encode all texts
    model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
    texts = [a["text"] for a in articles]
    embeddings = model.encode(texts, convert_to_tensor=True)
    keep = []
    for idx, emb in enumerate(embeddings):
        is_dup = False
        for j in keep:
            sim = util.cos_sim(emb, embeddings[j]).item()
            if sim > threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(idx)

    # Return deduped list preserving original order
    return [articles[i] for i in keep]

if __name__ == "__main__":
    raw = scrape_all(max_per_site=10)
    unique = dedupe_articles(raw)
    for art in unique:
        print(f"[{art['site']}] {art['title']}")
        print(art["url"])
        print(art["text"][:200], "…")
        print("-" * 80)
