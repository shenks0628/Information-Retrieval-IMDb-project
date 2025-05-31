import json
import re
import glob

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

keywords = ["Tom Cruise", "Ethan Hunt", "Ethan", "Hunt", "Tom", "Cruise"]
# keywords = ["Daniel Radcliffe", "Harry Potter", "Harry", "Potter", "Daniel", "Radcliffe"]

json_files = glob.glob("../data/Mission-Impossible*.json")
# json_files = glob.glob("../data/Harry-Potter*.json")

filter_complete_reviews = []
filter_split_reviews = []
movie_titles = set()
movie_ids = set()

for file in json_files:
    print(f"Processing file: {file}")
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 收集電影資訊
    movie_titles.add(data["metadata"].get("movie_title", ""))
    movie_ids.add(data["metadata"].get("movie_imdb_id", ""))
    # 篩選評論
    for group in data["reviews"].values():
        for review in group:
            text = (review.get("content", "")).lower()
            splitted_text = split_sentences(text)
            for sentence in splitted_text:
                if any(re.search(r'\b' + re.escape(k.lower()) + r'\b', sentence) for k in keywords):
                    filter_split_reviews.append(sentence)
                    if text not in filter_complete_reviews:
                        filter_complete_reviews.append(text)

output = {
    "metadata": {
        "movie_title": list(movie_titles),
        "movie_imdb_id": list(movie_ids)
    },
    "complete_reviews": filter_complete_reviews,
    "split_reviews": filter_split_reviews
}

output_file = 'data/TomCruise.json'
# output_file = 'data/DanielRadcliffe.json'

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print(f"Filtered {len(filter_complete_reviews)} complete reviews and {len(filter_split_reviews)} split reviews.")
print(f"Output saved to {output_file}")
