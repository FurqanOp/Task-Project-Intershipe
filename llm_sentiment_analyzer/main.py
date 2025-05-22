from sentiment import get_sentiment
print("imported Sentimentr")

if __name__ == "__main__":
    reviews = [
        "This product is amazing! It exceeded my expectations.",
        "It works okay, nothing special.",
        "Completely disappointed. Waste of money."
    ]

    for review in reviews:
        sentiment = get_sentiment(review)
        print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")
