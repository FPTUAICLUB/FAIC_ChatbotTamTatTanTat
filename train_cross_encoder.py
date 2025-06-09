from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from pyvi import ViTokenizer
import torch

# Load tokenizer
model_name = "cross-encoder/nli-deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dữ liệu huấn luyện với 3 nhãn (0, 1, 2)
data = [
    {"text": "Thông tin về Nguyễn Văn Hiệu", "label": 1},
    {"text": "Hãy cung cấp cho tôi một số bức ảnh về sự kiện AI lai rai", "label": 0},
    {"text": "ai là", "label": 0},
    {"text": "là ai", "label": 0},
    {"text": "ai sinh nhật ngày", "label": 0},
    {"text": "trưởng ban", "label": 0},
    {"text": "phó ban", "label": 0},
    {"text": "thành viên", "label": 0},
    {"text": "chức vụ", "label": 0},
    {"text": "họ tên", "label": 0},
    {"text": "chủ nhiệm", "label": 0},
    {"text": "phó chủ nhiệm", "label": 0},
    {"text": "ngày sinh", "label": 0},
    {"text": "mã số sinh viên", "label": 0},
    {"text": "mssv", "label": 0},
    {"text": "tên", "label": 0},
    {"text": "phỏng vấn gen", "label": 1},
    {"text": "AI LAI RAI", "label": 1},
    {"text": "AI LAI RAI HỘT MỨT", "label": 1},
    {"text": "Chúc tết", "label": 1},
    {"text": "Halloween", "label": 1},
    {"text": "Day En Rose", "label": 1},
    {"text": "NATHANIA", "label": 1},
    {"text": "Lookback fall", "label": 1},
    {"text": "Retro night", "label": 1},
    {"text": "Rivaro Adventure", "label": 1},
    {"text": "Fruity K-night", "label": 1},
    {"text": "Teambuilding", "label": 1},
    {"text": "Trung thu", "label": 1},
    {"text": "Hackathon", "label": 1},
    {"text": "Cuộc đua số", "label": 1},
    {"text": "F-Cup", "label": 1},
    {"text": "Sự kiện giải toán", "label": 1},
    {"text": "Sự kiện", "label": 1},
    {"text": "clb AI", "label": 2},
    {"text": "thành lập", "label": 2},
    {"text": "Câu lạc bộ AI thành lập như thế nào?", "label": 2}  # Nhãn mới
]

# Chuyển dữ liệu thành Dataset
dataset = Dataset.from_list(data)

# Hàm tiền xử lý với batch
def preprocess_function(examples):
    processed_texts = [ViTokenizer.tokenize(text) for text in examples["text"]]  # Tokenize tiếng Việt
    return tokenizer(processed_texts, truncation=True, padding="max_length", max_length=128)

# Tokenize dữ liệu
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Chuyển đổi dữ liệu thành tensor đúng cách
encoded_dataset = encoded_dataset.map(lambda e: {
    'input_ids': e['input_ids'],
    'attention_mask': e['attention_mask'],
    'labels': e['label']
}, batched=True)

# Load model với 3 nhãn
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Sử dụng DataCollator để tránh lỗi `FutureWarning`
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Cấu hình training
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=8,  # Chỉnh số epoch muốn train
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    logging_steps=1,  # Ghi log sau mỗi step
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",  # Không gửi log đến WandB
    disable_tqdm=False  # Bật thanh tiến trình
)


# Huấn luyện với Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset,
    data_collator=data_collator  
)

# Train
trainer.train()

# Lưu mô hình
trainer.save_model(training_args.output_dir)  
tokenizer.save_pretrained(training_args.output_dir)
