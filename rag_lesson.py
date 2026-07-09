import chromadb
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1 — Create Vector DB with company documents
client = chromadb.Client()
collection = client.create_collection(
    name="company_docs",
    embedding_function=None
)

# Add company policy documents
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
        [1.0, 0.1, 0.1, 0.1],  # leave policy
        [0.9, 0.1, 0.1, 0.1],  # sick leave
        [0.1, 1.0, 0.1, 0.1],  # salary
        [0.1, 0.9, 0.1, 0.1],  # bonus
        [0.1, 0.1, 1.0, 0.1],  # remote work
        [0.1, 0.1, 0.1, 1.0],  # overtime
    ],
    ids=["d1", "d2", "d3", "d4", "d5", "d6"]
)

# Step 2 — RAG function
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def rag_answer(question: str, query_embedding: list) -> str:
    # Retrieve relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    relevant_docs = results["documents"][0]
    context = "\n".join(relevant_docs)
    
    # Generate answer using retrieved context
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"""You are an HR assistant.
Answer ONLY based on these company documents:
{context}
If the answer is not in the documents, say 'I don't have that information'"""},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# Test RAG
print(rag_answer("How many days of annual leave do I get?", [1.0, 0.1, 0.1, 0.1]))
print("---")
print(rag_answer("What is the bonus policy?", [0.1, 0.9, 0.1, 0.1]))