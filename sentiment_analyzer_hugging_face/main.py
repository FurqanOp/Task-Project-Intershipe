from sentiment import get_sentiment

if __name__ == "__main__":
    reviews = [
        "I love this product!",
    "It is just a regular phone.",
    "This is a total waste of time"
    ]

    print("Running Sentiment Analysis...\n")

    for review in reviews:
        print(f"Review: {review}")
        sentiment = get_sentiment(review)
        print(f"Predicted Sentiment: {sentiment}\n")

