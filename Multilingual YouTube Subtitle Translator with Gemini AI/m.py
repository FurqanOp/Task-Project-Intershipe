from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import google.generativeai as genai

# Gemini API Setup
API_KEY = "AIzaSyB4pqXQsUPilvEEB7Ad-xvh4us0NX78gpE"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def fetch_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        print("[INFO] Available subtitle languages:")
        available_transcripts = list(transcript_list)
        for idx, t in enumerate(available_transcripts, 1):
            gen = " (auto-generated)" if t.is_generated else ""
            print(f"{idx}. {t.language} [{t.language_code}]{gen}")

        print("\nSelect the subtitle language you want to fetch:")
        for idx, t in enumerate(available_transcripts, 1):
            print(f"{idx}. {t.language} [{t.language_code}]")
        choice = int(input("Enter the number corresponding to the subtitle language: ")) - 1
        source_transcript = available_transcripts[choice]
        source_lang = source_transcript.language_code

        print("\nEnter the target language code for translation (e.g. 'en' for English, 'fr' for French).")
        print("Press Enter to keep the same language as the subtitle language.")
        target_lang = input("Enter target language code: ").strip() or source_lang

        if target_lang != source_lang and source_transcript.is_translatable:
            print(f"[INFO] Translating '{source_lang}' transcript to '{target_lang}'...")
            transcript = source_transcript.translate(target_lang).fetch()
        else:
            print(f"[INFO] Fetching transcript in '{source_lang}'...")
            transcript = source_transcript.fetch()

        text = " ".join([entry.text for entry in transcript])
        return text

    except TranscriptsDisabled:
        print("[ERROR] Transcripts are disabled for this video.")
    except Exception as e:
        print(f"[ERROR] {e}")
    return ""

def summarize_with_gemini(text):
    try:
        print("[INFO] Summarizing transcript with Gemini...")
        prompt = "Summarize the following YouTube transcript in English:\n\n" + text[:20000]  # Limit to fit token size
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[ERROR] Gemini summarization failed: {e}")
        return ""

# --- Main ---
if __name__ == "__main__":
    video_id = input("Enter YouTube Video ID: ").strip()
    transcript_text = fetch_transcript(video_id)

    if transcript_text:
        print("\n--- Transcript (Partial) ---\n")
        print(transcript_text[:1000] + "\n...")  # Show a preview
        summarize = input("\nDo you want to summarize this with Gemini? (y/n): ").strip().lower()
        if summarize == "y":
            summary = summarize_with_gemini(transcript_text)
            print("\n--- Gemini Summary ---\n")
            print(summary)
    else:
        print("[INFO] No transcript fetched or transcript is empty.")
