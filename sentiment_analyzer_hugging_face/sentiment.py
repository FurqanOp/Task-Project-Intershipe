from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from scipy.special import softmax

# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sentiment labels
labels = ['Negative', 'Neutral', 'Positive']

def get_sentiment(review: str) -> str:
    try:
        # Preprocessing
        encoded_input = tokenizer(review, return_tensors='pt', truncation=True)
        with torch.no_grad():
            output = model(**encoded_input)
        scores = output.logits[0].numpy()
        probs = softmax(scores)
        predicted_label = labels[np.argmax(probs)]

        return predicted_label
    except Exception as e:
        print("Error:", e)
        return "Error"
