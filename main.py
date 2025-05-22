from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

genai.configure(api_key="AIzaSyD_pnSqhpEKpgWo0ghMjYNcY_LLiM6UMkg")
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()

class TranslationRequest(BaseModel):
    sentence: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Zero-Shot Translation API. Use POST /translate to translate text."}

@app.post("/translate")
def translate_text(request: TranslationRequest):
    try:
        prompt = f"Translate the following sentence to English:\n\n{request.sentence}"
        response = model.generate_content(prompt)
        return {"translation": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

