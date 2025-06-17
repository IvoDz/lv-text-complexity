### python hf_demo.py --model bert --text "Šis ir diezgan vienkāršs teksts."  

import argparse
import torch
from transformers import (
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
    BertTokenizer, BertForSequenceClassification
)

label_map = {0: "viegls", 1: "vidējs", 2: "sarežģīts"}

def load_model(model_name):
    if model_name == "roberta":
        tokenizer = XLMRobertaTokenizer.from_pretrained("ivodz/roberta-lv-complexity", use_fast=False)
        model = XLMRobertaForSequenceClassification.from_pretrained("ivodz/roberta-lv-complexity")
    elif model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("ivodz/bert-lv-complexity")
        model = BertForSequenceClassification.from_pretrained("ivodz/bert-lv-complexity")
    else:
        raise ValueError("Model must be 'bert' or 'roberta'")
    return tokenizer, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["bert", "roberta"], help="Choose model: bert or roberta")
    parser.add_argument("--text", type=str, help="Input text")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)

    if args.text:
        text = args.text
    else:
        text = input("Ievadiet tekstu: ")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    print("Varbūtības (viegls, vidējs, sarežģīts):", probs.tolist()[0])
    predicted_class = torch.argmax(probs, dim=1).item()
    print("Sarežģītības klase:", label_map[predicted_class])

if __name__ == "__main__":
    main()
