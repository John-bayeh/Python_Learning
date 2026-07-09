import chromadb

# Create client
client = chromadb.Client()

# Create collection WITHOUT automatic embedding
collection = client.create_collection(
    name="employee_feedbacks",
    embedding_function=None
)

# Add with manual simple embeddings (just for learning)
collection.add(
    documents=[
        "I love working here, the team is amazing!",
        "My salary is too low and I need a raise",
        "The management is terrible and I want to quit",
        "Work life balance is great, I am happy here",
        "I am not getting paid enough for my work",
        "My manager never listens to my concerns",
    ],
    embeddings=[
        [1.0, 0.1, 0.1],  # positive
        [0.1, 1.0, 0.1],  # salary complaint
        [0.1, 0.1, 1.0],  # management complaint
        [0.9, 0.1, 0.1],  # positive
        [0.1, 0.9, 0.1],  # salary complaint
        [0.1, 0.1, 0.9],  # management complaint
    ],
    ids=["f1", "f2", "f3", "f4", "f5", "f6"]
)

# Search for salary complaints
results = collection.query(
    query_embeddings=[[0.1, 1.0, 0.1]],
    n_results=2
)

print("Searching for: salary complaints")
print("Found:")
for doc in results["documents"][0]:
    print(f"→ {doc}")