def build_prompt(review: str) -> str:
    return f"""
You are a helpful assistant trained to classify the sentiment of product reviews.
Classify the following review as Positive, Negative, or Neutral.

Review:
\"{review}\"

Sentiment:
"""
