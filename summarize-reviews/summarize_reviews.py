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
            with open(os.path.join(self.upload_dir, json_file), 'r') as f:
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