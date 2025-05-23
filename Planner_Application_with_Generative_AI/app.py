import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load Gemini API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

INTERVIEW_DOMAINS = [
    "Cybersecurity",
    "Data Science",
    "Gen AI",
    "Cloud",
    "DevOps",
    "Software Engineering"
]

def build_prompt(user_choice, sub_choice=None, duration=None, goal=None, domain=None, weak_areas=None):
    if user_choice == "Gym":
        return (
            f"You are a certified personal trainer. Create a {duration} workout planner focused on {sub_choice.lower()}. "
            f"The user's goal is: '{goal}'. Provide a weekly routine with exercises, sets, reps, and rest days."
        )
    elif user_choice == "Interview Prep":
        return (
            f"You are a senior career coach. The user is preparing for an interview in the domain: '{domain}'. "
            f"The goal is: '{goal}'. Generate a {duration} weekly interview preparation planner with learning topics, "
            f"mock interview sessions, and key resources tailored to '{domain}' domain."
        )
    elif user_choice == "Study":
        return (
            f"You are an academic coach. Create a {duration} study plan for a student whose goal is to score 90-100. "
            f"Focus especially on the weak areas: {weak_areas}. Break the plan into weekly segments with topics, revision tasks, and test practices."
        )
    return "Invalid input."

def call_gemini(prompt):
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API Error: {e}"

def main():
    st.title("🎯 Personalized Planner Generator")

    user_choice = st.radio("What would you like to plan?", ("Gym", "Interview Prep", "Study"))

    duration = st.selectbox("Select duration", ["1 week", "3 weeks", "4 weeks", "6 weeks"])

    sub_choice = domain = weak_areas = goal = ""

    if user_choice == "Gym":
        workout_type = st.radio("Choose your workout type:", ("Cardio", "Weight Training"))
        sub_choice = workout_type
        goal = st.text_input("What's your fitness goal?")

    elif user_choice == "Interview Prep":
        domain = st.selectbox("Choose your domain:", INTERVIEW_DOMAINS)
        goal = st.text_input("What's your goal? (e.g., Get into Google, Master the field)")

    elif user_choice == "Study":
        weak_areas = st.text_input("Enter your weak topics or subjects:")
        goal = "Score between 90 and 100"

    if st.button("Generate Planner"):
        if user_choice == "Gym" and (not goal):
            st.error("Please enter your fitness goal.")
            return
        if user_choice == "Interview Prep" and (not goal):
            st.error("Please enter your interview goal.")
            return
        if user_choice == "Study" and (not weak_areas):
            st.error("Please enter your weak topics or subjects.")
            return

        prompt = build_prompt(user_choice, sub_choice, duration, goal, domain, weak_areas)
        st.markdown("### 🚀 Sending Prompt to Gemini...")
        st.code(prompt)

        response = call_gemini(prompt)

        st.markdown("### 📅 Generated Planner:")
        st.write(response)

if __name__ == "__main__":
    main()
