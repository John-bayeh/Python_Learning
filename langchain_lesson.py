from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

history = []

def chat(user_input: str):
    history.append(HumanMessage(content=user_input))
    all_messages = [SystemMessage(content="You are a helpful HR assistant.")] + history
    response = llm.invoke(all_messages)
    history.append(AIMessage(content=response.content))
    return response.content

# Test memory
print(chat("My name is Yohannes"))
print("---")
print(chat("What is my name?"))  # Should remember "Yohannes"