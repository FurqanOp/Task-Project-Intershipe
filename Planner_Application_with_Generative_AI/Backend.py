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

# ✅ Updated to Gemini 1.5 Flash
def call_gemini(prompt):
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API Error: {e}"

def main():
    print("\n🎯 What would you like to plan?")
    print("1. Gym\n2. Interview Prep\n3. Study")
    choice_map = {"1": "Gym", "2": "Interview Prep", "3": "Study"}
    choice_input = input("Enter choice (1/2/3): ").strip()
    user_choice = choice_map.get(choice_input)

    if not user_choice:
        print("❌ Invalid choice. Exiting.")
        return

    duration = input("🕒 Enter duration (1 week / 3 weeks / 4 weeks / 6 weeks): ").strip()
    sub_choice = domain = weak_areas = goal = ""

    if user_choice == "Gym":
        print("🏋️ Choose your workout type:\n1. Cardio\n2. Weight Training")
        sub_input = input("Enter choice (1/2): ").strip()
        sub_choice = "Cardio" if sub_input == "1" else "Weight Training"
        goal = input("🎯 What's your fitness goal? ").strip()

    elif user_choice == "Interview Prep":
        print("📌 Choose your domain:")
        for i, dom in enumerate(INTERVIEW_DOMAINS, 1):
            print(f"{i}. {dom}")
        dom_input = input("Enter domain number: ").strip()
        if not dom_input.isdigit() or not (1 <= int(dom_input) <= len(INTERVIEW_DOMAINS)):
            print("❌ Invalid domain. Exiting.")
            return
        domain = INTERVIEW_DOMAINS[int(dom_input) - 1]
        goal = input("🎯 What's your goal? (e.g., Get into Google, Master the field) ").strip()

    elif user_choice == "Study":
        weak_areas = input("📚 Enter your weak topics or subjects: ").strip()
        goal = "Score between 90 and 100"

    prompt = build_prompt(user_choice, sub_choice, duration, goal, domain, weak_areas)

    print("\n🚀 Sending Prompt to Gemini...\n")
    print(f"--- PROMPT TEMPLATE ---\n{prompt}\n")

    response = call_gemini(prompt)
    print("\n📅 Generated Planner:\n")
    print(response)

if __name__ == "__main__":
    main()
