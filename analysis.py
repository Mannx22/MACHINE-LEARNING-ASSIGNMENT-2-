import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

SEED = 42
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

np.random.seed(SEED)
torch.manual_seed(SEED)

print("\nLoading IMDb dataset...")
dataset = load_dataset("imdb")

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]
test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

print(f"  Train size: {len(train_texts):,} | Test size: {len(test_texts):,}")
print(f"  Label distribution (train): {pd.Series(train_labels).value_counts().to_dict()}")

print("\nTraining Logistic Regression baseline...")

tfidf = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
    stop_words="english",
)
X_train_tfidf = tfidf.fit_transform(train_texts)
X_test_tfidf = tfidf.transform(test_texts)

lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, n_jobs=-1)
lr_model.fit(X_train_tfidf, train_labels)

lr_preds = lr_model.predict(X_test_tfidf)
lr_acc = accuracy_score(test_labels, lr_preds)
lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(
    test_labels, lr_preds, average="binary"
)
print(f"  LR Accuracy: {lr_acc:.4f} | Precision: {lr_p:.4f} | Recall: {lr_r:.4f} | F1: {lr_f1:.4f}")

print("\nPreparing DistilBERT tokeniser and dataset...")


class IMDbDataset(Dataset):
    """PyTorch Dataset wrapper for IMDb reviews."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

if DEVICE.type == "cpu":
    print("  WARNING: No GPU detected. Using 2,000 train / 500 test samples for speed.")
    print("  For full results, run on a GPU (e.g., Google Colab).")
    train_sample_texts = train_texts[:2000]
    train_sample_labels = train_labels[:2000]
    test_sample_texts = test_texts[:500]
    test_sample_labels = test_labels[:500]
else:
    train_sample_texts = train_texts
    train_sample_labels = train_labels
    test_sample_texts = test_texts
    test_sample_labels = test_labels

train_dataset = IMDbDataset(list(train_sample_texts), list(train_sample_labels), tokenizer, MAX_LEN)
test_dataset = IMDbDataset(list(test_sample_texts), list(test_sample_labels), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"\nFine-tuning DistilBERT for {EPOCHS} epoch(s)...")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Step {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"  → Epoch {epoch+1} complete. Avg train loss: {avg_loss:.4f}")

print("\nEvaluating DistilBERT on test set...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

bert_acc = accuracy_score(all_labels, all_preds)
bert_p, bert_r, bert_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="binary"
)
print(f"  BERT Accuracy: {bert_acc:.4f} | Precision: {bert_p:.4f} | Recall: {bert_r:.4f} | F1: {bert_f1:.4f}")

print("\n" + "="*65)
print("  MODEL COMPARISON RESULTS")
print("="*65)
results = pd.DataFrame({
    "Model": ["Logistic Regression (TF-IDF)", "DistilBERT (fine-tuned)"],
    "Accuracy": [f"{lr_acc:.4f}", f"{bert_acc:.4f}"],
    "Precision": [f"{lr_p:.4f}", f"{bert_p:.4f}"],
    "Recall": [f"{lr_r:.4f}", f"{bert_r:.4f}"],
    "F1-Score": [f"{lr_f1:.4f}", f"{bert_f1:.4f}"],
})
print(results.to_string(index=False))
print("="*65)

results.to_csv("results.csv", index=False)
print("\nResults saved to results.csv")

model.save_pretrained("./distilbert_imdb")
tokenizer.save_pretrained("./distilbert_imdb")
print("Fine-tuned model saved to ./distilbert_imdb/")

