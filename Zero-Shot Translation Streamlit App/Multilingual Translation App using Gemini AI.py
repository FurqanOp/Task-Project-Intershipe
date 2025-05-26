from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyD_pnSqhpEKpgWo0ghMjYNcY_LLiM6UMkg")
model = genai.GenerativeModel("gemini-1.5-flash")

# FastAPI app
app = FastAPI(title="Multilingual Translation API", version="2.0")

# Request body schema
class TranslationRequest(BaseModel):
    sentence: str
    source_lang: str  # e.g., "Spanish"
    target_lang: str  # e.g., "English"

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Multilingual Translation API. Use POST /translate with sentence, source_lang, and target_lang."
    }

@app.post("/translate")
def translate_text(request: TranslationRequest):
    try:
        prompt = (
            f"Translate the following sentence from {request.source_lang} to {request.target_lang}:\n\n"
            f"{request.sentence}"
        )
        response = model.generate_content(prompt)
        return {
            "source_language": request.source_lang,
            "target_language": request.target_lang,
            "original_text": request.sentence,
            "translation": response.text.strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

