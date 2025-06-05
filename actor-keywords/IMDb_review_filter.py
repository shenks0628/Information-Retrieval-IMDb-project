import json
import re
import glob

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

full_keywords = {
    "Tom Cruise": {
        "Mission-Impossible": ["Ethan Hunt", "Ethan", "Hunt", "Tom Cruise", "Tom", "Cruise"],
        "Top-Gun": ["Pete Mitchell", "Maverick", "Pete", "Mitchell", "Tom Cruise", "Tom", "Cruise"],
        "Edge-of-Tomorrow": ["Cage", "Tom Cruise", "Tom", "Cruise"],
        "Oblivion": ["Jack", "Tom Cruise", "Tom", "Cruise"],
    },
    "Daniel Radcliffe": {
        "Harry-Potter": ["Harry Potter", "Harry", "Potter", "Daniel Radcliffe", "Daniel", "Radcliffe"],
    },
    "Morgan Freeman": {
        "The-Shawshank-Redemption": ["Red", "Morgan Freeman", "Morgan", "Freeman"],
        "Batman-Begins": ["Lucius Fox", "Lucius", "Fox", "Morgan Freeman", "Morgan", "Freeman"],
        "The-Dark-Knight": ["Lucius Fox", "Lucius", "Fox", "Morgan Freeman", "Morgan", "Freeman"],
        "Oblivion": ["Beech", "Morgan Freeman", "Morgan", "Freeman"],
        "Now-You-See-Me": ["Thaddeus Bradley", "Thaddeus", "Bradley", "Morgan Freeman", "Morgan", "Freeman"],
        "London-Has-Fallen": ["President Trumbull", "Allan Trumbull", "Allan", "Trumbull", "Morgan Freeman", "Morgan", "Freeman"],
        "Angel-Has-Fallen": ["President Trumbull", "Allan Trumbull", "Allan", "Trumbull", "Morgan Freeman", "Morgan", "Freeman"],
    },
}

# actor = "Tom Cruise"
# actor = "Daniel Radcliffe"
actor = "Morgan Freeman"

output_file = f"data/{actor.replace(' ', '')}.json"

# json_files = glob.glob("../data/Mission-Impossible*.json") + glob.glob("../data/Top-Gun*.json") + glob.glob("../data/Edge-of-Tomorrow*.json") + glob.glob("../data/Oblivion*.json")
# json_files = glob.glob("../data/Harry-Potter*.json")
json_files = glob.glob("../data/The-Shawshank-Redemption*.json") + glob.glob("../data/Batman-Begins*.json") + glob.glob("../data/The-Dark-Knight*.json") + glob.glob("../data/Oblivion*.json") + glob.glob("../data/Now-You-See-Me*.json") + glob.glob("../data/London-Has-Fallen*.json") + glob.glob("../data/Angel-Has-Fallen*.json")

actor_keywords = full_keywords.get(actor, {})
movies = list(actor_keywords.keys())

filter_complete_reviews = []
filter_split_reviews = []
movie_titles = set()
movie_ids = set()

for file in json_files:
    print(f"Processing file: {file}")
    keywords = []
    for movie in movies:
        if movie in file:
            keywords = actor_keywords[movie]
            break
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    movie_titles.add(data["metadata"].get("movie_title", ""))
    movie_ids.add(data["metadata"].get("movie_imdb_id", ""))
    
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

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print(f"Filtered {len(filter_complete_reviews)} complete reviews and {len(filter_split_reviews)} split reviews.")
print(f"Output saved to {output_file}")
