from langchain_community.embeddings import HuggingFaceEmbeddings

class CustomE5Embedding(HuggingFaceEmbeddings):
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct"):
        super().__init__(model_name=model_name)
        
    def embed_documents(self, texts):
        """
        Embed a list of documents using the instruction format
        with sentiment analysis instruction
        """
        formatted_texts = [f"instruct: Determine the sentiment of the following text.\nquery: {t}" for t in texts]
        return super().embed_documents(formatted_texts)
    
import os, json
import numpy as np
from langchain_community.vectorstores import FAISS

class embeddingDB:
    def __init__(self, db_dir="faiss_db", upload_dir="data", model_name="intfloat/multilingual-e5-large-instruct"):
        self.db_dir = db_dir
        self.upload_dir = upload_dir
        self.embedding_model = CustomE5Embedding(model_name=model_name)
        # os.makedirs(db_dir, exist_ok=True)
        # os.makedirs(upload_dir, exist_ok=True)

    def update_faiss_db_json(self):
        # json_files = [f for f in os.listdir(self.upload_dir) if f.endswith(".json")]
        json_files = ["sentiment-test.json"]
        for json_file in json_files:
            with open(os.path.join(self.upload_dir, json_file), 'r') as f:
                data = json.load(f)

            reviews = data["reviews"]

            x = [review["content"] for review in reviews]
            y = [review["class"] for review in reviews]

            contents = []
            metadata = []
            for idx in range(len(x)):
                contents.append(x[idx])
                meta = {"id": idx, "y_test": y[idx]}
                metadata.append(meta)

            print("Data Extracted")
            
            vectorstore = FAISS.from_texts(
                contents, 
                self.embedding_model,
                metadatas=metadata
            )
            
            index_path = os.path.join(self.db_dir, "self-test")
            print(f"Saving data to {index_path}...")
            vectorstore.save_local(index_path)

    def update_faiss_db_npz(self):
        # npzs = [f for f in os.listdir(self.upload_dir) if f.endswith(".npz")]
        npzs = ["imdb_converted.npz"]
        for npz in npzs:
            data = np.load(os.path.join(self.upload_dir, npz), allow_pickle=True)
            x_train = data['x_train']
            y_train = data['y_train']
            x_test = data['x_test']
            y_test = data['y_test']
            
            train_contents = []
            train_metadata = []
            for idx, x in enumerate(x_train):
                train_contents.append(x)
                meta = {"id": idx, "y_train": y_train[idx]}
                train_metadata.append(meta)
            
            test_contents = []
            test_metadata = []
            for idx, x in enumerate(x_test):
                test_contents.append(x)
                meta = {"id": idx, "y_test": y_test[idx]}
                test_metadata.append(meta)
            
            print("Data Extracted")
            
            train_vectorstore = FAISS.from_texts(
                train_contents,
                self.embedding_model,
                metadatas=train_metadata
            )
            train_index_path = os.path.join(self.db_dir, "train")
            print(f"Saving train data to {train_index_path}...")
            train_vectorstore.save_local(train_index_path)
            
            test_vectorstore = FAISS.from_texts(
                test_contents,
                self.embedding_model,
                metadatas=test_metadata
            )
            test_index_path = os.path.join(self.db_dir, "test")
            print(f"Saving test data to {test_index_path}...")
            test_vectorstore.save_local(test_index_path)

    def load_local_db(self, is_train=True, is_self=False):
        index_path = os.path.join(self.db_dir, "train" if is_train else ("self-test" if is_self else "test"))
        db = FAISS.load_local(
            folder_path=index_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )
        
        docs = list(db.docstore._dict.values())
        docs.sort(key=lambda doc: doc.metadata["id"])

        x_embedding = db.index.reconstruct_n(0, db.index.ntotal)
        x_text = []
        y = []
        for doc in docs:
            x_text.append(doc.page_content)
            y.append(doc.metadata["y_train" if is_train else "y_test"])
            
        return np.array(x_embedding), np.array(x_text), np.array(y)

