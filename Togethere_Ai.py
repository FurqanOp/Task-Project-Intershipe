import requests

# API Key and headers
TOGETHER_API_KEY = "98b097d0c98e74b1f6ddb1825845c44b448cbe351c176f89a4bf35884c41c898"
headers = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}

# Base prompt
BASE_PROMPT = """
You are an expert sentiment classifier.

Your task is to classify the sentiment of the following product review into one of the following:
- Positive
- Negative
- Neutral

Respond with only the sentiment label.

Review: """

# Sentiment analyzer
def analyze_sentiment(review):
    full_prompt = BASE_PROMPT + f'"{review}"\nSentiment:'

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "prompt": full_prompt,
        "temperature": 0.3,
        "max_tokens": 5,
        "top_p": 0.9,
        "stop": ["\n"]
    }

    response = requests.post("https://api.together.xyz/inference", headers=headers, json=payload)
    result = response.json()

    # Debugging: print the full response if 'output' is missing
    if "output" not in result:
        print("Error response from API:", result)
        return "Error"

    sentiment = result["output"]["choices"][0]["text"].strip()
    return sentiment

# Sample reviews
reviews = [
    "I absolutely love this product!",
    "It's okay, not too bad but not great.",
    "Terrible experience, I want a refund."
]

# Run classification
for review in reviews:
    sentiment = analyze_sentiment(review)
    print(f"Review: {review}\nSentiment: {sentiment}\n")
