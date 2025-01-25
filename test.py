import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "./saved_model"

# 加载 tokenizer 和 model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# 让模型使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 确保输入在相同设备上
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return "Positive" if predicted_class == 1 else "Negative"

# 测试
print(predict("This movie is fantastic! I love it."))
print(predict("I hate this movie. It was so bad."))