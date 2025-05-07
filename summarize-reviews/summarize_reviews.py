from langchain_community.embeddings import HuggingFaceEmbeddings

class CustomE5Embedding(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")

#------------------------------------------------------------------------

import os, json
import numpy as np
from sklearn.cluster import KMeans
from langchain_community.vectorstores import FAISS

class KMeansReviewClustering:
    def __init__(self, k=5, db_dir="faiss_db", upload_dir="uploaded_docs", model_name="intfloat/multilingual-e5-large-instruct"):
        self.k = k
        self.db_dir = db_dir
        self.upload_dir = upload_dir
        self.embedding_model = CustomE5Embedding(model_name=model_name)
        os.makedirs(db_dir, exist_ok=True)
        os.makedirs(upload_dir, exist_ok=True)

    def update_faiss_db(self):
        json_files = [f for f in os.listdir(self.upload_dir) if f.endswith(".json")]
        for json_file in json_files:
            with open(os.path.join(self.upload_dir, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)

            for score_str, reviews in data["reviews"].items():
                contents = [review["content"] for review in reviews]
                print(f"Score {score_str}: {len(contents)} reviews")
                vectorstore = FAISS.from_texts(contents, self.embedding_model)
                print(f"Saving to {os.path.join(self.db_dir, f'{json_file[:-5]}_{score_str}')}")
                vectorstore.save_local(os.path.join(self.db_dir, f"{json_file[:-5]}_{score_str}"))

    def kmeans_clustering(self, vectors):
        kmeans = KMeans(n_clusters=self.k, random_state=42)
        kmeans.fit(vectors)

        centers = kmeans.cluster_centers_

        closest_indices = []
        for i in range(self.k):
            center = centers[i]
            distances = np.linalg.norm(vectors - center, axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)

        return closest_indices

    def get_kmeans_reviews(self, movie_title, review_score):
        # 從本地載入 FAISS 向量資料庫
        db_path = os.path.join(self.db_dir, f"{movie_title}_{review_score}")
        db = FAISS.load_local(
            folder_path=db_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )

        # 取得所有 Document 物件
        docs = list(db.docstore._dict.values())

        # 取得所有向量
        vectors = db.index.reconstruct_n(0, db.index.ntotal)

        # 進行 k-means 並取得最貼近中心的索引
        closest_indices = self.kmeans_clustering(vectors)

        # 根據索引從 docstore 裡取出對應的評論文字
        reviews = [docs[i].page_content.replace("\n", " ") for i in closest_indices]
        return reviews

#------------------------------------------------------------------------

import os
from openai import OpenAI

class MovieReviewSummarizer:
    def __init__(self, gemini_api_key):
        self.api_key = gemini_api_key
        self.model = "gemini-2.0-flash"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        self.system_prompt = """
            你是個專業的電影評論分析師，專長是閱讀多則關於同一部電影的評論，並從中提煉出最核心的觀點與整體評價。你的任務是接收一系列電影評論文字，並生成一段精簡、客觀、全面的摘要。

            你的摘要必須：
            1.  **使用繁體中文** 撰寫。這是最重要的要求。
            2.  **反映整體情緒**：準確傳達評論者對電影的普遍看法（例如：壓倒性好評、褒貶不一、多數負評）。
            3.  **點出關鍵優缺點**：總結評論中反覆提及的主要優點（如精彩的演技、驚人的視覺效果、深刻的劇情）和缺點（如拖沓的節奏、薄弱的角色、不合邏輯的情節）。
            4.  **提及討論焦點**：如果評論集中討論某些特定面向（例如：導演風格、配樂、主題深度），應在摘要中提及。
            5.  **保持中立客觀**：僅根據提供的評論內容進行總結，不加入個人意見或外部資訊。
            6.  **語言精練易懂**：摘要應簡潔明瞭，避免冗長或使用過於專業的術語（除非評論普遍使用）。

            請專注於綜合分析所有提供的評論，輸出一份高品質的繁體中文摘要。
        """
        self.prompt_template = """
            請根據以下多則關於電影「{movie_title}」的評論，生成一段符合要求的繁體中文摘要。

            **評論列表：**

            {reviews_list}

            **繁體中文摘要：**
        """

        # 將 API Key 設定為環境變數
        os.environ["OPENAI_API_KEY"] = self.api_key

        # 建立 OpenAI 用戶端
        self.client = OpenAI(
            base_url=self.base_url
        )

    def summarize_reviews(self, movie_title, movie_reviews):
        retrieved_chunks = "\n".join(movie_reviews)
        final_prompt = self.prompt_template.format(
            movie_title=movie_title, 
            reviews_list=retrieved_chunks
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": final_prompt},
            ]
        )
        return response.choices[0].message.content