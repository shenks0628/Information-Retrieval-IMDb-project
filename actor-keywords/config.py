BASE_STOP_WORDS = [
    "movie", "film", "book", "mission", "impossible", "one", "like", "scene",
    "time", "really", "plot", "story", "see", "first", "part", "get", "much",
    "make", "also", "even", "mi", "watch", "still", "could", "way", "go", "think",
    "know", "lot", "would", "end", "seen", "new", "say", "made", "every", "never",
    "two", "woo", "thing", "feel", "10", "come", "act", "keep", "take", "read",
    "give", "fan", "leave", "many", "find", "little", "well", "seem", "year",
    "director", "show", "people", "though", "look", "last", "bite", "enjoy",
    "half", "want", "start", "back", "play", "actor", "actress", "cast", "knight"
]

ACTOR_SPECIFIC_STOP_WORDS = {
    "TomCruise": [
        "tom cruise", "ethan hunt", "ethan", "hunt", "pete mitchell", "pete",
        "mitchell", "maverick", "cage", "jack", "tom", "cruise"
    ],
    "DanielRadcliffe": [
        "daniel radcliffe", "harry potter", "harry", "potter", "daniel", "radcliffe"
    ],
    "MorganFreeman": [
        "morgan freeman", "red", "redding", "lucius fox", "lucius", "fox",
        "morgan", "freeman", "beech", "thaddeus bradley", "thaddeus", "bradley",
        "president trumbull", "allan trumbull", "allan", "trumbull", "andy",
        "robbins", "shawshank", "bale", "tim", "caine", "batman", "joker",
        "michael", "bruce", "nolan", "wayne", "gary", "alfred"
    ]
}

DEFAULT_TOP_K = 20
MAX_FEATURES = 10000