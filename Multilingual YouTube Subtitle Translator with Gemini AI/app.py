import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import google.generativeai as genai

# Gemini API Setup
API_KEY = "AIzaSyAyp1J83E8RdD4jYCWc-bGbAuuJUUGLe_s"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def fetch_transcript(video_id, selected_idx, target_lang):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_transcripts = list(transcript_list)

        source_transcript = available_transcripts[selected_idx]
        source_lang = source_transcript.language_code

        transcript = None
        if target_lang != source_lang and source_transcript.is_translatable:
            try:
                transcript = source_transcript.translate(target_lang).fetch()
            except Exception:
                transcript = source_transcript.fetch()
        else:
            transcript = source_transcript.fetch()

        text = " ".join([entry.text for entry in transcript])
        return text, available_transcripts
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except Exception as e:
        st.error(f"Error: {e}")
    return "", []

def summarize_with_gemini(text):
    try:
        prompt = "Summarize the following YouTube transcript in English:\n\n" + text[:20000]
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini summarization failed: {e}")
        return ""

def main():
    st.title("YouTube Transcript Fetcher & Gemini Summarizer")

    video_id = st.text_input("Enter YouTube Video ID", "")

    if video_id:
        # Try to list available transcripts
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available_transcripts = list(transcript_list)

            languages = [
                f"{t.language} [{t.language_code}]{' (auto-generated)' if t.is_generated else ''}"
                for t in available_transcripts
            ]

            selected_idx = st.selectbox("Select subtitle language", options=range(len(available_transcripts)), format_func=lambda x: languages[x])

            source_lang = available_transcripts[selected_idx].language_code
            target_lang = st.text_input(
                "Enter target language code for translation (leave empty to keep original)",
                value=source_lang,
                max_chars=10,
            ).strip() or source_lang

            if st.button("Fetch Transcript"):
                with st.spinner("Fetching transcript..."):
                    transcript_text, _ = fetch_transcript(video_id, selected_idx, target_lang)
                    if transcript_text:
                        st.subheader("Transcript (Partial)")
                        st.text_area("Transcript Text", value=transcript_text[:3000] + ("..." if len(transcript_text) > 3000 else ""), height=300)

                        if st.button("Summarize with Gemini"):
                            with st.spinner("Summarizing transcript with Gemini..."):
                                summary = summarize_with_gemini(transcript_text)
                                if summary:
                                    st.subheader("Gemini Summary")
                                    st.write(summary)
                    else:
                        st.info("No transcript found or transcript is empty.")

        except TranscriptsDisabled:
            st.error("Transcripts are disabled for this video.")
        except Exception as e:
            st.error(f"Failed to retrieve transcript languages: {e}")

if __name__ == "__main__":
    main()
