!pip install transformers datasets accelerate evaluate scikit-learn pandas

import pandas as pd
import numpy as np
import torch from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.head())

train["label"] = train["label"].map({"positive": 1 , "negative":0})

# from sklearn.model_selection import train_test_split
# train_df , valid_df = train_test_split(train, test_size = 0.1 , random_state = 7 , stratify = train["label"])

from datasets import Dataset

# training_data = Dataset.from_pandas(train_df)
training_data = Dataset.from_pandas(train)
# val_data = Dataset.from_pandas(valid_df)
test_data = Dataset.from_pandas(test)

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True, padding="max_length", max_length=512)
token_train = training_data.map(tokenize_function, batched=True)
# token_val = val_data.map(tokenize_function, batched=True)
token_test = test_data.map(tokenize_function, batched=True)

print(token_train[0])
print(token_test[0])

# token_train = token_train.remove_columns(["Id", "review", "__index_level_0__"])
token_train = token_train.remove_columns(["Id", "review"])
# token_val = token_val.remove_columns(["Id", "review", "__index_level_0__"])
token_test = token_test.remove_columns(["Id", "review"])

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny",num_labels = 2)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=7,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    logging_steps=100,
    eval_strategy="no",
    save_strategy="epoch",
    # load_best_model_at_end=True
)

from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=token_train,
    # eval_dataset=token_val,
    compute_metrics = compute_metrics
)

trainer.train()

# trainer.evaluate()

predictions = trainer.predict(token_test)
pred_labels = np.argmax(predictions.predictions, axis=1)
label_map = {1: "positive", 0: "negative"}
test["label"] = [label_map[p] for p in pred_labels]
test[["Id", "label"]].to_csv("submission.csv", index=False)

from google.colab import files
files.download("submission.csv")

