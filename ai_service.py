from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from data_analyzer import get_company_summary
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

def ask_ai(question: str, history: list = []) -> str:
    try:
        company_data = get_company_summary()

        chat_history = []
        for msg in history[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                chat_history.append(AIMessage(content=msg["content"]))

        chat_history.append(HumanMessage(content=question))

        all_messages = [
         SystemMessage(content=f"""You are a professional HR assistant 
for an Ethiopian company. You have access to real company data:
{company_data}

IMPORTANT RULES:
- If a user tells you their name or role in conversation, remember it
- Prioritize information from the conversation over company data
- Company data is for employee queries, not for identifying who is talking to you
- Be concise and professional.""")
        ] + chat_history

        response = llm.invoke(all_messages)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"