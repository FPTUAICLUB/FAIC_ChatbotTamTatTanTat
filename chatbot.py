import faiss
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from embedding import get_embedding
from faiss_index_members import load_faiss_db

faiss_index, docs = load_faiss_db()
llm = ChatOpenAI(api_key="", model_name="gpt-4o-mini")

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Dựa trên thông tin thành viên sau đây: {context}, hãy trả lời câu hỏi: {question}"
)

def search_faiss(query, top_k=300):
    query_embedding = get_embedding(query).reshape(1, -1)
    D, I = faiss_index.search(query_embedding, top_k)
    
    results = [docs[i].page_content for i in I[0] if i < len(docs)]
    return " | ".join(results) if results else "Không tìm thấy thông tin phù hợp."

def chatbot_interface(user_input):
    context = search_faiss(user_input)
    if context == "Không tìm thấy thông tin phù hợp.":
        return context
    response = llm.predict(prompt_template.format(context=context, question=user_input))
    return response
