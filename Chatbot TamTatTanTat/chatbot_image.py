import base64
from io import BytesIO
from PIL import Image
from fetch_data_events import df
from faiss_index_events import load_faiss_db
from embedding import get_embedding
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

def decode_base64_image(image_base64):
    if image_base64:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        return image
    return None

def search_faiss(query, top_k=15):
    """Tìm kiếm dữ liệu trong FAISS và trả về danh sách hình ảnh & mô tả"""
    query_embedding = get_embedding(query).astype("float32").reshape(1, -1)
    D, I = faiss_index.search(query_embedding, top_k)

    descriptions = []
    image_list = []

    for idx in I[0]:
        if idx == -1:
            continue
        matched_data = docs[idx]
        metadata = matched_data["metadata"]
        
        # Giải mã hình ảnh từ Base64
        image = decode_base64_image(metadata.get("image_base64"))

        descriptions.append(f"📌 {metadata['name']} - {metadata['description']}")
        if image:
            image_list.append(image)

    if not descriptions:
        return "Không tìm thấy thông tin phù hợp.", []

    return "\n\n".join(descriptions), image_list

faiss_index, docs = load_faiss_db()
llm = ChatOpenAI(model_name="gpt-4o-mini")

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Bạn là một chatbot AI thông minh. Dưới đây là dữ liệu về sự kiện có liên quan:

    {context}

    Người dùng hỏi: "{question}"
    
    Hãy trả lời câu hỏi của người dùng một cách chi tiết và dễ hiểu.
    """
)

def generate_openai_response(user_input, context):
    """Gọi OpenAI API để tạo câu trả lời dựa trên dữ liệu FAISS"""
    prompt = prompt_template.format(context=context, question=user_input)
    response = llm.predict(prompt)
    return response

def chatbot_interface(user_input):
    """Hàm chính của chatbot"""
    context, images = search_faiss(user_input)
    
    if context == "Không tìm thấy thông tin phù hợp.":
        return images, "Xin lỗi, tôi không tìm thấy thông tin phù hợp."

    response = generate_openai_response(user_input, context)
    return response, images
