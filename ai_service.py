from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from data_analyzer import get_company_summary
from dotenv import load_dotenv
import chromadb
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# Setup Vector DB with company policies
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="company_policies",
    embedding_function=None
)

collection.add(
    documents=[
        "Annual leave policy: All employees get 14 working days per year",
        "Sick leave policy: Employees get 30 days paid sick leave per year",
        "Salary review happens every January based on performance score",
        "Employees with score above 8.5 get 15% bonus annually",
        "Remote work is allowed 2 days per week for Engineering department",
        "Overtime pay is 1.5x normal rate for hours above 8 per day",
    ],
    embeddings=[
        [1.0, 0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1, 0.1],
        [0.1, 1.0, 0.1, 0.1],
        [0.1, 0.9, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.1],
        [0.1, 0.1, 0.1, 1.0],
    ],
    ids=["d1", "d2", "d3", "d4", "d5", "d6"]
)

def get_relevant_docs(question: str) -> str:
    # Simple keyword matching for embedding selection
    q = question.lower()
    if any(w in q for w in ["leave", "vacation", "annual"]):
        embedding = [1.0, 0.1, 0.1, 0.1]
    elif any(w in q for w in ["sick", "medical", "ill"]):
        embedding = [0.9, 0.1, 0.1, 0.1]
    elif any(w in q for w in ["salary", "pay", "compensation", "raise"]):
        embedding = [0.1, 1.0, 0.1, 0.1]
    elif any(w in q for w in ["bonus", "reward", "incentive"]):
        embedding = [0.1, 0.9, 0.1, 0.1]
    elif any(w in q for w in ["remote", "work from home", "wfh"]):
        embedding = [0.1, 0.1, 1.0, 0.1]
    elif any(w in q for w in ["overtime", "extra hours"]):
        embedding = [0.1, 0.1, 0.1, 1.0]
    else:
        embedding = [0.5, 0.5, 0.5, 0.5]

    results = collection.query(
        query_embeddings=[embedding],
        n_results=2
    )
    return "\n".join(results["documents"][0])

def ask_ai(question: str, history: list = []) -> str:
    try:
        company_data = get_company_summary()
        relevant_docs = get_relevant_docs(question)

        chat_history = []
        for msg in history[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                chat_history.append(AIMessage(content=msg["content"]))

        chat_history.append(HumanMessage(content=question))

        all_messages = [
            SystemMessage(content=f"""You are a professional HR assistant 
for an Ethiopian company. You have access to:

EMPLOYEE DATA:
{company_data}

RELEVANT COMPANY POLICIES:
{relevant_docs}

Answer accurately using both the employee data and policies above.
Be concise and professional.""")
        ] + chat_history

        response = llm.invoke(all_messages)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"