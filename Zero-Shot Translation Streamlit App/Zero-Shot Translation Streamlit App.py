import streamlit as st
import requests

# Streamlit UI
st.set_page_config(page_title="Zero-Shot Translator", page_icon="🌍")
st.title("🌍 Zero-Shot Translation Tool")
st.markdown("Translate any sentence from any language into English using Gemini 1.5 Flash.")

# Input
sentence = st.text_input("Enter a sentence in any language:", "")

if st.button("Translate"):
    if sentence.strip() == "":
        st.warning("Please enter a sentence to translate.")
    else:
        try:
            # ✅ Correct API endpoint
            response = requests.post(
                "http://127.0.0.1:8000/translate",
                json={"sentence": sentence}
            )

            if response.status_code == 200:
                translation = response.json()["translation"]
                st.success("Translation:")
                st.write(translation)
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {str(e)}")
