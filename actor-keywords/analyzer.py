from sklearn.feature_extraction.text import TfidfVectorizer
import config

class KeywordAnalyzer:
    def __init__(self, max_features=config.MAX_FEATURES):
        self.vectorizer = TfidfVectorizer(
            stop_words=None,  # 停用詞已經在 Preprocessor 處理過了
            max_features=max_features, 
            sublinear_tf=True
        )
        self.terms = None
        self.scores = None
        self.sorted_idx = None

    def fit_transform(self, documents):
        """計算 TF-IDF 並排序"""
        X = self.vectorizer.fit_transform(documents)
        self.scores = X.sum(axis=0).A1
        self.terms = self.vectorizer.get_feature_names_out()
        
        # 根據分數進行降序排序
        self.sorted_idx = self.scores.argsort()[::-1]

    def get_top_k_keywords(self, k=config.DEFAULT_TOP_K):
        """取得前 k 個關鍵字及其分數"""
        if self.terms is None or self.scores is None:
            raise ValueError("Analyzer has not been fitted yet. Call fit_transform first.")
            
        top_indices = self.sorted_idx[:k]
        return [(self.terms[i], self.scores[i]) for i in top_indices]