import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import google.generativeai as genai
import fitz  # PyMuPDF

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Gemini
genai.configure(api_key="AIzaSyCXzW-oQeyQYkhKozrn5IGwgfBFHYg3zV8")  # Replace with your Gemini 1.5 Flash API key

# 🔹 Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 🔹 NLTK preprocessing
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return list(set(tokens))

# 🔹 Gemini prompt generation
def prompt_gemini_model(jd_text, resume_text):
    prompt = f"""
You are an AI assistant that compares job descriptions and resumes.

1. Extract relevant skills from both the JD and resume.
2. Match them semantically (even if phrased differently).
3. Return:
   - Matched skills
   - Missing skills (required in JD but not found in resume)
   - A match score from 0 to 100.

Job Description:
{jd_text}

Resume:
{resume_text}

Respond in this JSON format:
{{
  "matched_skills": [...],
  "missing_skills": [...],
  "match_score": %
}}
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# 🔹 Main function
def compare_resume_with_jd(jd_path, resume_path):
    jd_raw = extract_text_from_pdf(jd_path)
    resume_raw = extract_text_from_pdf(resume_path)

    jd_clean = ' '.join(preprocess(jd_raw))
    resume_clean = ' '.join(preprocess(resume_raw))

    result = prompt_gemini_model(jd_clean, resume_clean)
    return result

# ✅ Example usage
if __name__ == "__main__":
    job_description_pdf = r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\v\JD_RC_Specialist_Engg.pdf"
    resume_pdf = r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\v\Furqan_Resume_Cybersecurity.pdf.pdf"

    output = compare_resume_with_jd(job_description_pdf, resume_pdf)
    print(output)
