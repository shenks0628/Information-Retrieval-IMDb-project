from article_embedding import embeddingDB

db = embeddingDB()
db.update_faiss_db_npz()
db.update_faiss_db_json()