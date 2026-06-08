from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME

client = Groq(api_key=GROQ_API_KEY)

def ask_ai(question: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role":"system","content":"""You are a professioanl HR assistant for ethiopian companies.I have a website called HR system with Employee management
-                       Salary and benefits questions
-                       Leave and attendance policies
-                        Ethiopian labor law questions
Be concise, professional, and friendly'"""},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
# temporary test - remove later
if __name__ == "__main__":
    result = ask_ai("What is class in python")
    print(result)