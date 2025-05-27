import nltk
import json
import os
import re
import fitz  # PyMuPDF
import google.generativeai as genai
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Download NLTK data (run once)
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")

# ✅ Configure Gemini
# IMPORTANT: Replace with your actual Gemini API key and ensure it's kept secure.
# Consider using environment variables for API keys in production.
try:
    # Replace with your Gemini API key
    genai.configure(api_key="AIzaSyAskmfS--W3hXFonkCEo5P0sMAiccaGwrE") # <<< IMPORTANT: SET YOUR API KEY HERE
except Exception as e:
    print(f"Error configuring Gemini: {e}. Please ensure your API key is correctly set.")
    exit()

# 🔹 Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

# 🔹 NLP Preprocessing
def preprocess(text):
    if not text:
        return []
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return list(set(tokens)) # Return unique lemmatized tokens

# 🔹 Enhanced Gemini Prompt for Individual Resume Analysis
def prompt_gemini_model(jd_text, resume_text):
    prompt = f"""
You are an AI assistant that compares job descriptions and resumes. Your goal is to identify the best candidate.

Job Description (Keywords):
{jd_text}

Resume (Keywords):
{resume_text}

Instructions:
1.  Extract relevant skills from both the JD and resume.
2.  Analyze projects and experience evidence in the resume text. Identify matches to core responsibilities implied by the JD keywords.
    IMPORTANT: Look for conceptual matches and transferable skills, even if exact keywords from the JD are missing in the resume.
3.  Prioritize meaningful contributions and relevant experience.
4.  In 'strong_areas', specifically highlight instances where the candidate's experience or projects strongly align with JD responsibilities, even without direct keyword overlap. Explain briefly WHY it's a strong conceptual match.
5.  Provide a 'match_score' as a float between 0.0 and 1.0, where 1.0 is a perfect match.
6.  Return ONLY valid JSON with these keys:
    -   matched_skills (list of strings)
    -   missing_skills (list of strings)
    -   strong_areas (list of strings, with explanations for conceptual matches)
    -   match_score (float)

Respond only in strict JSON format (no markdown, no explanations outside the JSON structure):
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Or your preferred model
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API for individual analysis: {e}")
        return "" # Return empty string or error indicator

# 🔹 Gemini Prompt for Comparative Narrative
def prompt_gemini_comparison_narrative(jd_text_keywords, new_candidate_name, new_candidate_analysis_json, all_candidates_data):
    previous_best_candidate_name = None
    previous_best_candidate_score = -1.0
    previous_best_candidate_analysis_str = "This is the first candidate being evaluated, or no other candidates have a positive score yet."

    # Find the best among *other* candidates
    if len(all_candidates_data) > 1 or (len(all_candidates_data) == 1 and new_candidate_name not in all_candidates_data):
        for name, data_dict in all_candidates_data.items():
            if name == new_candidate_name: # Don't compare candidate to themselves from the main list
                continue

            if isinstance(data_dict, dict):
                score_str = data_dict.get("match_score", "0.0")
                try:
                    score = float(score_str)
                except ValueError:
                    score = 0.0
                
                if score > previous_best_candidate_score:
                    previous_best_candidate_score = score
                    previous_best_candidate_name = name
                    previous_best_candidate_analysis_str = json.dumps({
                        "name": name,
                        "match_score": score,
                        "matched_skills": data_dict.get("matched_skills", []),
                        "missing_skills": data_dict.get("missing_skills", []),
                        "strong_areas": data_dict.get("strong_areas", [])
                    }, indent=2)
    
    # If after checking all_candidates_data, only the new_candidate_name (implicitly) was found as best
    # or no one else had a better score, the default message remains.
    # If a previous best was found:
    if previous_best_candidate_name:
        pass # previous_best_candidate_analysis_str is already set
    elif len(all_candidates_data) <= 1 and new_candidate_name in all_candidates_data : # Only the new candidate exists
         previous_best_candidate_analysis_str = "This is the first candidate being evaluated."


    prompt = f"""
You are an AI recruitment assistant. Your task is to provide a qualitative assessment of a new candidate and compare them to the previously best-evaluated candidate (if any) for a specific job.

Job Description (Keywords):
{jd_text_keywords}

New Candidate Under Review:
Name: {new_candidate_name}
Individual Analysis (JSON):
{json.dumps(new_candidate_analysis_json, indent=2)}

Previously Best Evaluated Candidate (if any):
{previous_best_candidate_analysis_str}

Instructions:
1.  Review the new candidate's analysis ({new_candidate_name}) against the Job Description keywords.
    -   Is this candidate a good fit for the job based on their analysis?
    -   Pay close attention to their 'strong_areas'. Do their projects and experiences show good alignment with the JD's core responsibilities, especially if they demonstrate transferable skills or conceptual understanding even without exact keyword matches? Elaborate on this.
2.  If there was a 'Previously Best Evaluated Candidate' provided (and it's different from the new candidate):
    -   Compare {new_candidate_name} with this previously best-evaluated candidate.
    -   Considering the JD, which candidate appears to be a stronger fit NOW?
    -   Explain your reasoning. Focus on the quality of experience, relevance of projects, identified strong areas, and how well their skills (matched and missing) align with the job's priorities. Do not solely rely on the 'match_score'.
3.  If {new_candidate_name} is effectively the first candidate being seriously considered (i.e., no valid previous best), provide an initial assessment of their suitability and potential for the role based on their analysis.

Provide your response as a concise, analytical paragraph or two. Do not use JSON for this output.
Be objective and highlight both strengths and areas for concern in your comparison.
Example if comparing:
"Assessing {new_candidate_name}: This candidate presents a compelling profile because... [details on JD fit, projects based on their analysis]. Their strong areas in [...] are particularly noteworthy. Compared to [Previous Best Candidate Name], {new_candidate_name} seems [stronger/weaker/comparable] because... [reasoning, focusing on qualitative aspects from their analyses]."
Example if first candidate:
"Assessing {new_candidate_name} (effectively the first evaluated or new top candidate): They demonstrate potential for this role due to... [details on JD fit, projects from their analysis]. Key strengths from their analysis include... However, areas like their missing skills in [...] might need further exploration or development."
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API for comparative narrative: {e}")
        return "Error: Could not generate comparative narrative."

# 🔹 Store candidate data in local JSON file
def store_candidate_summary(candidate_name, summary_json, filename="resume_db.json"):
    db = {}
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                db = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {filename} contains invalid JSON. Starting with an empty database.")
            db = {}
            
    db[candidate_name] = summary_json

    try:
        with open(filename, "w") as f:
            json.dump(db, f, indent=2)
    except IOError as e:
        print(f"Error writing to {filename}: {e}")


# 🔹 Find best candidate numerically (fixed)
def find_best_candidate_numerically(filename="resume_db.json"):
    if not os.path.exists(filename):
        return "Database file not found. No candidates to compare."

    try:
        with open(filename, "r") as f:
            db = json.load(f)
    except json.JSONDecodeError:
        return "Error: Could not decode JSON from the database file."
        
    if not db:
        return "No candidates stored in the database yet."

    best_candidate_name = None
    best_score = -1.0 

    for name, data in db.items():
        if isinstance(data, dict):
            score_value = data.get("match_score", 0.0) # Default to 0.0 if missing
            try:
                # Ensure score is float
                score = float(score_value)
            except (ValueError, TypeError):
                print(f"Warning: Could not parse match_score '{score_value}' for candidate {name}. Treating as 0.0.")
                score = 0.0
        else:
            print(f"Warning: Data for candidate {name} is not a dictionary. Skipping.")
            continue

        if score > best_score:
            best_score = score
            best_candidate_name = name

    if best_candidate_name is not None:
        return best_candidate_name, best_score
    else:
        return "No candidate with a positive score found, or all scores were invalid."

# 🔹 Main function
def compare_resume_with_jd(jd_path, resume_path, candidate_name):
    print(f"\nProcessing Candidate: {candidate_name}...")
    jd_raw = extract_text_from_pdf(jd_path)
    resume_raw = extract_text_from_pdf(resume_path)

    if not jd_raw:
        print(f"Could not extract text from Job Description: {jd_path}")
        return
    if not resume_raw:
        print(f"Could not extract text from Resume: {resume_path}")
        return

    jd_clean_keywords = ' '.join(preprocess(jd_raw))
    resume_clean_keywords = ' '.join(preprocess(resume_raw))

    if not jd_clean_keywords:
        print("Job Description text resulted in no usable keywords after preprocessing.")
        return
    if not resume_clean_keywords:
        print(f"Resume for {candidate_name} resulted in no usable keywords after preprocessing.")
        return

    print("Generating individual analysis...")
    gemini_response_json_str = prompt_gemini_model(jd_clean_keywords, resume_clean_keywords)

    if not gemini_response_json_str:
        print("Failed to get a response from Gemini for individual analysis.")
        return

    current_candidate_analysis_json = None
    try:
        # Attempt to find JSON block more robustly
        match = re.search(r'```json\s*(\{.*?\})\s*```', gemini_response_json_str, re.DOTALL)
        if match:
            cleaned_json_str = match.group(1)
        else:
            # Fallback to finding the first '{' and last '}'
            json_match = re.search(r'\{.*\}', gemini_response_json_str, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON block found in Gemini output for individual analysis.")
            cleaned_json_str = json_match.group(0)
        
        current_candidate_analysis_json = json.loads(cleaned_json_str)

    except Exception as e:
        print("⚠️ Gemini response for individual analysis is not valid JSON or could not be extracted. Raw output:")
        print(gemini_response_json_str)
        print(f"\nError: {e}")
        return

    # Ensure match_score is float in the stored JSON
    try:
        current_candidate_analysis_json['match_score'] = float(current_candidate_analysis_json.get('match_score', 0.0))
    except (ValueError, TypeError):
        print(f"Warning: Could not convert match_score '{current_candidate_analysis_json.get('match_score')}' to float. Defaulting to 0.0.")
        current_candidate_analysis_json['match_score'] = 0.0
    
    store_candidate_summary(candidate_name, current_candidate_analysis_json)
    print(f"\n✅ Resume for '{candidate_name}' analyzed and individual analysis stored.")
    print("Individual Analysis JSON:")
    print(json.dumps(current_candidate_analysis_json, indent=2))

    # Load all candidates for comparison narrative
    all_candidates_data = {}
    db_filename = "resume_db.json"
    if os.path.exists(db_filename):
        try:
            with open(db_filename, "r") as f:
                all_candidates_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not load {db_filename} for comparison, file might be corrupted.")
            # all_candidates_data will remain empty, so new candidate will be treated as first.
    
    print("\n🔍 Generating Comparative Assessment by Gemini...")
    comparative_narrative = prompt_gemini_comparison_narrative(
        jd_clean_keywords,
        candidate_name,
        current_candidate_analysis_json,
        all_candidates_data
    )
    print("\n--- Gemini's Comparative Assessment ---")
    print(comparative_narrative)
    print("--- End of Assessment ---")

    # Display the numerically best candidate from the updated database
    best_candidate_info = find_best_candidate_numerically(db_filename)
    if isinstance(best_candidate_info, tuple): # (name, score)
        print(f"\n🏆 Numerically Best Candidate in DB: {best_candidate_info[0]} with score {best_candidate_info[1]:.2f}")
    else: # It's a message string
        print(f"\n{best_candidate_info}")


# ✅ Example usage
if __name__ == "__main__":
    # --- IMPORTANT: SET YOUR FILE PATHS HERE ---
    job_description_pdf = r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\k\JD_RC_Specialist_Engg.pdf" # <<< CHANGE THIS
    resume_pdf_candidate1 = r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\k\Furqan Ahmed (CV).pdf.pdf" # <<< CHANGE THIS
    candidate1_name = "Ahmed"

    # resume_pdf_candidate2 = r"C:\path\to\another\resume.pdf" # <<< For a second candidate
    # candidate2_name = "Jane Doe"

    # --- Make sure NLTK data is downloaded (run these lines once if needed) ---
    # print("Checking NLTK data...")
    # try:
    #     word_tokenize("test")
    # except LookupError:
    #     nltk.download("punkt")
    # try:
    #     stopwords.words('english')
    # except LookupError:
    #     nltk.download("stopwords")
    # try:
    #     WordNetLemmatizer().lemmatize("test")
    # except LookupError:
    #     nltk.download("wordnet")
    # print("NLTK data check complete.")
    # --------------------------------------------------------------------------
    
    # For the first candidate:
    if not os.path.exists(job_description_pdf):
        print(f"Error: Job Description PDF not found at: {job_description_pdf}")
    elif not os.path.exists(resume_pdf_candidate1):
        print(f"Error: Resume PDF not found for {candidate1_name} at: {resume_pdf_candidate1}")
    else:
        compare_resume_with_jd(job_description_pdf, resume_pdf_candidate1, candidate1_name)

    # # Example for processing a second candidate (uncomment and update paths)
    # print("\n" + "="*50 + "\n") # Separator
    # if not os.path.exists(resume_pdf_candidate2):
    #    print(f"Error: Resume PDF not found for {candidate2_name} at: {resume_pdf_candidate2}")
    # else:
    #    compare_resume_with_jd(job_description_pdf, resume_pdf_candidate2, candidate2_name)