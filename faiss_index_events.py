import os
import faiss
import numpy as np
import pandas as pd
from fetch_data_events import df  
from embedding import get_embedding

FAISS_INDEX_PATH = 'faiss_index_image'



def create_faiss_db(df):
    docs = []
    embeddings = []
    
    for _, row in df.iterrows():
        doc = {
            "page_content": f"id: {row.id}, name: {row.name}, description: {row.description}",
            "metadata": {
                "id": row.id,
                "name": row.name,
                "description": row.description,
                "image_base64": row.image_base64
            }
        }
        docs.append(doc)
        embeddings.append(get_embedding(doc["page_content"]))

    # Tạo FAISS index
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

    return faiss_index, docs



def load_faiss_db():
    """
    Tải FAISS index từ file nếu có, nếu không thì tạo mới.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        print("Đang tải FAISS index từ file...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        docs = []

        for _, row in df.iterrows():
            doc = {
                "page_content": f"id: {row.id}, name: {row.name}, description: {row.description}",
                "metadata": {
                    "id": row.id,
                    "name": row.name,
                    "description": row.description,
                    "image_base64": row.image_base64
                }
            }
            docs.append(doc)
        return faiss_index, docs
    else:
        print("Không tìm thấy FAISS index, tạo mới...")
        return create_faiss_db(df)
