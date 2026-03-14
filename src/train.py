import json
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# ==========================================================
# TRANSFORMER TRAINING SCRIPT: DistilBERT FOR CRISIS LABELS
# ==========================================================

def load_preprocessed_reddit_data(json_path: str):
    """
    Loads Reddit JSON that has already passed through the Presidio Anonymizer
    and has been labelled with a 'crisis_label' / severity bucket.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract only valid strings and their labels
    texts = []
    labels = []
    
    # Map high=2, medium=1, low=0
    label_map = {"high": 2, "medium": 1, "low": 0}
    
    for item in data:
        text = item.get("post_text") or item.get("comment_text", "")
        # Only use anonymized, labelled data
        if item.get("preprocessing_status") == "anonymized" and item.get("crisis_label") in label_map:
            texts.append(text)
            labels.append(label_map[item["crisis_label"]])
            
    return {"text": texts, "label": labels}

def train_distilbert_crisis_model():
    print("1. Loading Tokenizer and Model...")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    # We are classifying into 3 buckets: low(0), medium(1), high(2)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    
    # For demonstration, we load mock data here:
    print("2. Mapping Dataset...")
    # data_dict = load_preprocessed_reddit_data("../data/reddit_cleaned.json")
    
    # Mock data directly for the script to run cleanly offline
    data_dict = {
        "text": [
            "I'm feeling so happy and positive today!",
            "I'm stressed but I'll manage.",
            "I can't take this anymore, I want to end everything right now."
        ],
        "label": [0, 1, 2]
    }
    raw_dataset = Dataset.from_dict(data_dict)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True)
    # Usually you split into train/eval...
    train_dataset = tokenized_dataset
    
    print("3. Defining Training Arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,              # 3 epochs for fine-tuning
        per_device_train_batch_size=8,   # standard batch size
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=10
    )
    
    print("4. Starting Sub-Model Training Loop...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # trainer.train() # Uncomment to run full PyTorch loop
    print("Training Complete! The weights can now be saved.")
    
    # Save the custom fine-tuned DistilBERT 
    # model.save_pretrained("../models/mindmate-distilbert-crisis")
    # tokenizer.save_pretrained("../models/mindmate-distilbert-crisis")
    
if __name__ == "__main__":
    train_distilbert_crisis_model()
