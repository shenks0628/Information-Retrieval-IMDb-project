# IMDb Reviews — Information Retrieval Project

This repository contains tools for crawling, processing, analyzing, and summarizing IMDb user reviews. It was built as a small information-retrieval / NLP project and includes components for crawling reviews, extracting actor-related keywords, sentiment analysis, and generating summary embeddings / FAISS indices.

## Repository Structure

- `IMDb_crawler.py` — Selenium-based crawler to fetch user reviews for a movie by IMDb ID and save them to `data/` as JSON.
- `data/` — Collected movie review JSON files (one file per movie).
- `actor-keywords/` — Tools to extract actor-related keywords from reviews:
  - `main.py`, `analyzer.py`, `preprocessor.py`, `data_loader.py`, `visualizer.py`, `IMDb_review_filter.py`, `config.py`
  - `actor-keywords/data/` contains example actor-specific JSON files.
- `sentiment-analysis/` — Sentiment analysis utilities and vector database updates (Keras/FAISS based):
  - `main.py`, `sentiment_analysis.py`, `article_embedding.py`, `keras_IMDb_converter.py`, `update_faiss_db.py`
  - `sentiment-analysis/data/`, `sentiment-analysis/models/`, `sentiment-analysis/faiss_db/` contains training data, trained model data and FAISS indices.
- `summarize-reviews/` — Review summarization pipeline that:
  - `main.py`, `summarize_reviews.py`, `update_faiss_db.py`
  - `summarize-reviews/faiss_db`, `summarize-reviews/summarize_reviews`, and `summarize-reviews/uploaded_docs` contains FAISS indices, summarize results, and reviews to be summarized.

## Quickstart

Prerequisites:

- Python 3.8+ (3.10/3.11 recommended)
- Install required packages (suggested):

```bash
python -m pip install -U pip
pip install selenium numpy pandas faiss-cpu tensorflow keras scikit-learn transformers sentence-transformers wordcloud matplotlib openai
```

Notes:
- The crawler uses Selenium and requires a compatible Chrome/WebDriver. You can use `webdriver-manager` to simplify driver management or configure your system ChromeDriver.
- Some modules (sentiment, summarization) depend on trained models and FAISS indices already included in the repo; heavy dependencies (TensorFlow, FAISS) are required only if you plan to re-train or regenerate indices.

## Usage

1) Crawl reviews for a movie

```bash
python IMDb_crawler.py --id ttXXXXXXXX --threads 4
# Example: crawl multiple IDs
python IMDb_crawler.py --id tt15398776 tt0123456 --threads 4
```

Options:
- `--id` — one or more IMDb IDs (required)
- `--threads` — number of parsing threads (default: 4)
- `--test_mode` — limit crawling for faster testing

2) Actor keywords extraction

Run the scripts in `actor-keywords/main.py` to preprocess reviews, filter actor-related reviews, extract keywords, and visualize results. See `actor-keywords/config.py` for configurable parameters.

3) Sentiment analysis

Run `sentiment-analysis/main.py` to run sentiment-related flows (embedding, converting, or updating FAISS DB). The `models/` and `faiss_db/` folders contain example trained models and indices.

4) Summarize reviews

Run `summarize-reviews/update_faiss_db.py` to build/update FAISS indices from JSON files placed in `summarize-reviews/uploaded_docs/` (one movie JSON per file). This creates per-movie, per-score FAISS vectorstores under `summarize-reviews/faiss_db/{Movie_Title}_{score}`.

Run `summarize-reviews/main.py` to extract representative reviews and generate summaries for each ratings (1-10). You might need to add/change the API_KEY information and LLM model to ensure the code works.

## Data format

- Movie JSONs saved by the crawler live in `data/` and follow the structure:

```json
{
  "metadata": {"movie_title": "...", "movie_imdb_id": "tt...", "reviews_count": 123},
  "reviews": {"10": [{"title":"...","content":"..."}]}
}
```

## Recommendations & Tips

- When running the crawler, prefer running on a machine with Chrome and an up-to-date ChromeDriver. Consider using `--test_mode` while developing.
- If you only need to run analysis/summarization and not re-crawl, you can skip installing `selenium` and the WebDriver.
- FAISS indices can be memory-intensive; use `faiss-cpu` vs GPU builds according to your environment.
