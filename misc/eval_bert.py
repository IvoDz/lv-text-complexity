import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

df = pd.read_csv("data/data.csv")
label_map = {"viegls": 0, "vidējs": 1, "sarežģīts": 2}
df["level"] = df["level"].map(label_map)

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["level"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["level"], random_state=42)

test_texts = test_df["text"].tolist()
test_labels = test_df["level"].tolist()
label_names = ["viegls", "vidējs", "sarežģīts"]

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx]),
        }

def evaluate(model_name):
    """ Prints sklearn classification reports for pretrained BERT models from HF hub """
    if model_name == "roberta":
        tokenizer = XLMRobertaTokenizer.from_pretrained("ivodz/roberta-lv-complexity", use_fast=False)
        model = XLMRobertaForSequenceClassification.from_pretrained("ivodz/roberta-lv-complexity")
    elif model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("ivodz/bert-lv-complexity")
        model = BertForSequenceClassification.from_pretrained("ivodz/bert-lv-complexity")
    else:
        raise ValueError("Unknown model name")

    dataset = TextDataset(test_texts, test_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds, gold = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            preds.extend(predictions.cpu().numpy())
            gold.extend(labels.cpu().numpy())

    print(f"\nClassification report for {model_name}:")
    print(classification_report(gold, preds, target_names=label_names, digits=3))

evaluate("bert")
evaluate("roberta")
