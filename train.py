import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from evaluate import load as load_metric
from torch.nn import CrossEntropyLoss

#  设定参数
MODEL_NAME = "bert-base-uncased"
SAVE_PATH = "./saved_model"
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

#  加载 tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


#  加载数据集
def load_data():
    train_dataset = load_dataset("imdb", split="train").shuffle(seed=42).select(range(3000))
    test_dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(500))

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    return train_dataset, test_dataset


train_dataset, test_dataset = load_data()

#  加载 Bert 模型
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

#  计算类别权重
num_pos = sum(1 for x in train_dataset if "positive" in x["text"].lower())
num_neg = len(train_dataset) - num_pos
class_weights = torch.tensor([num_neg / len(train_dataset), num_pos / len(train_dataset)]).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu"))


#  自定义损失函数
def custom_loss(output, labels):
    loss_fct = CrossEntropyLoss(weight=class_weights)
    return loss_fct(output.logits, labels)


#  训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=200,
    weight_decay=0.01,
    learning_rate=LEARNING_RATE,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
)

#  计算准确率
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


#  创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

#  训练模型
trainer.train()
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)


#  预测函数
def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return "Positive" if predicted_class == 1 else "Negative"


#  运行测试
print(predict("This movie is fantastic! I love it."))
print(predict("I hate this movie. It was so bad."))