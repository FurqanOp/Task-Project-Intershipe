import streamlit as st
from sentiment import get_sentiment

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("🧠 Sentiment Analysis App")
st.write("Enter a sentence and classify it as Positive, Neutral, or Negative.")

# Input text
review = st.text_area("Enter your sentence:", "")

if st.button("Analyze Sentiment"):
    if review.strip():
        sentiment = get_sentiment(review)
        st.subheader("Predicted Sentiment:")
        st.success(sentiment)
    else:
        st.warning("Please enter some text to analyze.")
