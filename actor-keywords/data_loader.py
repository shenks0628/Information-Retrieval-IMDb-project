import json
import os

def load_reviews_from_json(file_path, kind_of_review):
    """
    從 JSON 檔案讀取指定類型的評論
    :param file_path: JSON 檔案路徑
    :param kind_of_review: 評論類別鍵值 (例如 'complete_reviews')
    :return: 評論列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get(kind_of_review, [])