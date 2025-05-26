import streamlit as st
import requests

# Streamlit UI Setup
st.set_page_config(page_title="Multilingual Translator", page_icon="🌐")
st.title("🌐 Multilingual Translator using Gemini 1.5 Flash")
st.markdown("Translate text from any language to another using a zero-shot LLM approach.")

# Input fields
sentence = st.text_area("Enter the sentence you want to translate:")
source_lang = st.text_input("Source Language (e.g., Spanish, Hindi, French):")
target_lang = st.text_input("Target Language (e.g., English, Arabic, German):")

# Translate button
if st.button("Translate"):
    if not sentence.strip() or not source_lang.strip() or not target_lang.strip():
        st.warning("Please fill in all fields.")
    else:
        try:
            payload = {
                "sentence": sentence,
                "source_lang": source_lang,
                "target_lang": target_lang
            }
            # Make POST request to FastAPI backend
            response = requests.post("http://127.0.0.1:8000/translate", json=payload)

            if response.status_code == 200:
                result = response.json()
                st.success("Translation Result:")
                st.write(f"**From ({source_lang}) → To ({target_lang})**")
                st.markdown(f"> {result['translation']}")
            else:
                st.error(f"Error from server: {response.json().get('detail')}")
        except Exception as e:
            st.error(f"Could not connect to the backend: {e}")
