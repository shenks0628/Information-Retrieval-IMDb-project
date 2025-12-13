import os
import config
from data_loader import load_reviews_from_json
from preprocessor import TextPreprocessor
from analyzer import KeywordAnalyzer
from visualizer import plot_wordcloud

def get_stop_words_for_file(file_name):
    """根據檔名決定要使用哪些額外的停用詞"""
    combined_stop_words = set(config.BASE_STOP_WORDS)
    base_name = os.path.basename(file_name)
    for actor, specific_words in config.ACTOR_SPECIFIC_STOP_WORDS.items():
        if actor in base_name:
            combined_stop_words.update(specific_words)
            
    return combined_stop_words

def main():
    file_names = [
        "data/TomCruise.json", 
        "data/DanielRadcliffe.json", 
        "data/MorganFreeman.json"
    ]
    
    kind_of_reviews = ["complete_reviews", "split_reviews"]
    
    for file_name in file_names:
        if not os.path.exists(file_name):
            print(f"Skipping {file_name}: File not found.")
            continue

        print(f"Processing file: {file_name}")
        
        # 1. 準備停用詞
        current_stop_words = get_stop_words_for_file(file_name)
        
        # 2. 初始化處理器
        preprocessor = TextPreprocessor(additional_stop_words=current_stop_words)
        analyzer = KeywordAnalyzer()
        
        for kind in kind_of_reviews:
            print(f"  Review Type: {kind}")
            
            # 3. 讀取資料
            reviews = load_reviews_from_json(file_name, kind)
            if not reviews:
                print("    No reviews found.")
                continue
                
            # 4. 前處理
            print("    Preprocessing reviews...")
            clean_docs = [preprocessor.process(review) for review in reviews]
            
            # 5. 分析
            print("    Analyzing keywords...")
            analyzer.fit_transform(clean_docs)
            
            # 6. 輸出前 20 個關鍵字
            top_20 = analyzer.get_top_k_keywords(20)
            print("    Top 20 Keywords:")
            for word, score in top_20:
                print(f"      {word}: {score:.2f}")
            
            # 7. 視覺化 (使用前 100 個關鍵字)
            top_100 = analyzer.get_top_k_keywords(100)
            plot_title = f"{os.path.basename(file_name)} - {kind}"
            plot_wordcloud(top_100, title=plot_title)
            
            print("-" * 50)

if __name__ == "__main__":
    main()