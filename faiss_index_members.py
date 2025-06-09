import faiss
import numpy as np
import os
from langchain.schema import Document
from embedding import get_embedding
from fetch_data_members import df
FAISS_INDEX_PATH = 'faiss_index.bin'

def create_faiss_db(df):
    docs = [
        Document(
            page_content=f"{row.ho_ten},gen:{row.gen}, MSSV: {row.mssv}, Giới tính: {row.gioi_tinh}, "
                         f"Ngày sinh: {row.ngay_sinh}, Ban: {row.ban}, "
                         f"Chức vụ: {row.chuc_vu}, Facebook: {row.link_fb}",
            metadata={"ho_ten": row.ho_ten}
        )
        for _, row in df.iterrows()
    ]

    embeddings = np.array([get_embedding(doc.page_content) for doc in docs])

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, FAISS_INDEX_PATH) 

    return faiss_index, docs

def load_faiss_db():
    if os.path.exists(FAISS_INDEX_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        docs = [
            Document(
                page_content=f"{row.ho_ten},gen:{row.gen}, MSSV: {row.mssv}, Giới tính: {row.gioi_tinh}, "
                             f"Ngày sinh: {row.ngay_sinh}, Ban: {row.ban}, "
                             f"Chức vụ: {row.chuc_vu}, Facebook: {row.link_fb}",
                metadata={"ho_ten": row.ho_ten}
            )
            for _, row in df.iterrows()
        ]
        return faiss_index, docs
    else:
        return create_faiss_db(df)