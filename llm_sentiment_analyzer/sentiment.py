import os
import httpx
from dotenv import load_dotenv
#from prompt_template import build_prompt

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def build_prompt(review: str) -> str:
    return f"""
You are a helpful assistant trained to classify the sentiment of product reviews.
Classify the following review as Positive, Negative, or Neutral.

Review:
\"{review}\"

Sentiment:
"""

def get_sentiment(review: str, model: str = "llama3-8b-8192") -> str:
    prompt = build_prompt(review)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a sentiment classification assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        response = httpx.post(GROQ_ENDPOINT, headers=HEADERS, json=payload, timeout=10)
        print("API Response:", response.text) 
        response.raise_for_status()
        result = response.json()
        sentiment = result['choices'][0]['message']['content'].strip().lower()

        # Normalize sentiment
        if "positive" in sentiment:
            return "Positive"
        elif "negative" in sentiment:
            return "Negative"
        elif "neutral" in sentiment:
            return "Neutral"
        else:
            return "Unknown"

    except Exception as e:
        print("Error:", e)
        return "Error"
