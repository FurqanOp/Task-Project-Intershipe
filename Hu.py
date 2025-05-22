from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect
from transformers import pipeline

app = FastAPI(title="Zero-Shot Translation API with Hugging Face", version="2.0")

# Input model
class TranslationRequest(BaseModel):
    sentence: str
    target_language: str  # e.g., 'en', 'fr', 'hi', etc.

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the HuggingFace Translation API. Use POST /translate to translate text."}

# Translation endpoint
@app.post("/translate")
def translate_text(request: TranslationRequest):
    try:
        # Detect source language
        source_lang = detect(request.sentence)

        # Construct model name, e.g., "Helsinki-NLP/opus-mt-hi-en"
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{request.target_language}"

        try:
            translator = pipeline("translation", model=model_name)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Model not found for {source_lang} to {request.target_language}.")

        result = translator(request.sentence)
        translation = result[0]["translation_text"]
        return {
            "source_language": source_lang,
            "target_language": request.target_language,
            "translation": translation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

