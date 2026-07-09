from transformers import pipeline

# Load a pretrained sentiment analysis model
# This downloads the model automatically (first time only)
classifier = pipeline("sentiment-analysis")

# Test with employee feedback examples
feedbacks = [
    "I love working here, the team is amazing!",
    "The management is terrible and I want to quit.",
    "Salary is okay but workload is too much.",
    "Best company I have ever worked for!"
]

for feedback in feedbacks:
    result = classifier(feedback)
    print(f"Feedback: {feedback}")
    print(f"Sentiment: {result[0]['label']} (confidence: {result[0]['score']:.2%})")
    print()