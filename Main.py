import gradio as gr
import openai
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from fetch_data_members import df as df_members
from fetch_data_events import df as df_events
from faiss_index_members import load_faiss_db as load_faiss_members
from faiss_index_events import load_faiss_db as load_faiss_events
from chatbot import chatbot_interface as chatbot_Thanhvien
from chatbot_image import chatbot_interface as chatbot_Sukien


model_name = "./checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

LABELS = [0, 1, 2]

faiss_index_members, docs_members = load_faiss_members()
faiss_index_events, docs_events = load_faiss_events()

def classify_input(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Kiểm tra kích thước logits
    if outputs.logits.shape[1] != len(LABELS):
        raise ValueError(f"Số lượng lớp mô hình ({outputs.logits.shape[1]}) không khớp với số nhãn trong LABELS ({len(LABELS)})")

    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_index = scores.argmax().item()

    predicted_label = LABELS[predicted_index]

    if predicted_label == 2:
        return "Không chắc chắn, vui lòng cung cấp thêm thông tin."
    
    return predicted_label

def multi_chatbot(user_input):
    category = classify_input(user_input)
    
    if category == 0:
        return chatbot_Thanhvien(user_input),None
    if category == 1:
        return chatbot_Sukien(user_input)
    else:
        return chatbot_Thanhvien(user_input),None

iface = gr.Interface(
    fn=multi_chatbot,
    inputs=gr.Textbox(label = 'Input',lines=2, placeholder="Nhập câu hỏi của bạn..."),
    outputs=[gr.Textbox(label = 'Output'),gr.Gallery(label="Hình ảnh")],
    title="Chatbot Tám Tất Tần Tật",
)

iface.launch()