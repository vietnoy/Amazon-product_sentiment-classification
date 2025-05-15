import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def load_model(model_dir="saved_model/distilbert_model"):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer

def predict(texts, model, tokenizer):
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.tolist(), probs.tolist()

if __name__ == "__main__":
    model, tokenizer = load_model()
    sample_texts = [
        "This product is amazing! Highly recommend.",
        "Worst customer service I have ever experienced."
    ]
    predictions, probabilities = predict(sample_texts, model, tokenizer)
    for text, pred, prob in zip(sample_texts, predictions, probabilities):
        label = "Positive" if pred == 1 else "Negative"
        print(f"Text: {text}\nPrediction: {label}, Probability: {prob}\n")