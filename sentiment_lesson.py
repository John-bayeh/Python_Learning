from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_sentiment(feedback: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": """Analyze the sentiment of the text. 
             Respond ONLY in this format:
             SENTIMENT: POSITIVE/NEGATIVE/NEUTRAL
             CONFIDENCE: 0-100%
             REASON: one sentence"""},
            {"role": "user", "content": feedback}
        ]
    )
    return response.choices[0].message.content

# Test with employee feedbacks
feedbacks = [
    "I love working here, the team is amazing!",
    "The management is terrible and I want to quit.",
    "Salary is okay but workload is too much.",
]

for feedback in feedbacks:
    print(f"Feedback: {feedback}")
    print(analyze_sentiment(feedback))
    print("---")