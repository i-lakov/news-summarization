import streamlit as st
from scraping import scrape_all, dedupe_articles
from summarize import summarize_bg

st.set_page_config(
    page_title="BG News Summarizer",
    layout="wide"
)

st.sidebar.title("Parameters")
max_per_site = st.sidebar.slider("Articles per site", 1, 10, 5)
dedupe_thr  = st.sidebar.slider("Dedup threshold", 0.70, 0.95, 0.85, 0.01)
en_min      = st.sidebar.slider("EN summary min length", 20, 200, 80)
en_max      = st.sidebar.slider("EN summary max length", en_min, 400, 200)
bg_max      = st.sidebar.slider("BG summary max length", 50, 400, 200)

st.title("Bulgarian News Summarizer 🔎📰")
st.write("""
Enter one or more RSS feed URLs (one per line), or just click **Run** to summarize
all configured sites.
""")

rss_input = st.text_area(
    "RSS URLs (override defaults):",
    value="",
    height=80
)

if st.button("Run summarization"):
    # Decide which feeds to scrape
    if rss_input.strip():
        feeds = [r.strip() for r in rss_input.split("\n") if r.strip()]
        # Override SITES temporarily
        from scraping import SITES
        SITES.clear()
        for idx, feed in enumerate(feeds, start=1):
            SITES[f"custom{idx}"] = {
                "rss": feed,
                "title_sel": SITES["bnt"]["title_sel"],
                "body_sel":  SITES["bnt"]["body_sel"]
            }

    st.info("Scraping articles…")
    raw = scrape_all(max_per_site)
    st.success(f"Fetched {len(raw)} articles.")

    st.info("Deduplicating…")
    unique = dedupe_articles(raw, threshold=dedupe_thr)
    st.success(f"{len(unique)} unique articles.")

    for art in unique:
        st.markdown("---")
        st.subheader(f"[{art['site']}] {art['title']}")
        with st.expander("Original text"):
            st.write(art["text"][:1000] + ("…" if len(art["text"])>1000 else ""))

        out = summarize_bg(
            art["text"],
            en_min_length=en_min,
            en_max_length=en_max,
            bg_max_length=bg_max
        )
        st.write("**EN summary:**", out["en_summary"])
        st.write("**BG summary:**", out["bg_summary"])
