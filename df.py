from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
import requests

app = FastAPI()

# List of valid models (you can expand this)
valid_models = {
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("so", "en"): "Helsinki-NLP/opus-mt-so-en",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    # Add more pairs as needed
}

class TranslationRequest(BaseModel):
    sentence: str
    target_language: str  # ISO code like "en", "hi", "fr"
@app.get("/")
def root():
    return {"message": "Translation API is running. Use POST /translate"}


@app.post("/translate")
def translate(request: TranslationRequest):
    try:
        src_lang = detect(request.sentence)
        tgt_lang = request.target_language.lower()

        model_key = (src_lang, tgt_lang)

        if model_key not in valid_models:
            raise HTTPException(status_code=400, detail=f"No translation model available for {src_lang} to {tgt_lang}")

        model_name = valid_models[model_key]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        inputs = tokenizer(request.sentence, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        return {
            "detected_language": src_lang,
            "target_language": tgt_lang,
            "translation": translated_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
