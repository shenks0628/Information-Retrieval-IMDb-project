import json
from summarize_reviews import MovieReviewSummarizer, KMeansReviewClustering

api_key = "gemini_api_key"  # 替換為你的 Gemini API 金鑰
summarizer = MovieReviewSummarizer(gemini_api_key=api_key)
kmeans_cluster = KMeansReviewClustering(k=5)

def summarize_all_reviews(movie_title):
    all_summaries = {}
    for score in range(1, 11):
        movie_reviews = kmeans_cluster.get_kmeans_reviews(movie_title, score)
        summary = summarizer.summarize_reviews(movie_title, movie_reviews)
        all_summaries[str(score)] = summary
        print(f"Score {score}: {summary}")
    
    with open(f"{movie_title}_summaries.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=4)
    print(f"Summaries saved to {movie_title}_summaries.json")

summarize_all_reviews("movie_title") # 替換為你要處理的電影名稱