# Imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def semantic_search(sources, query):
    # Load a lightweight pre-trained model
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    # Encode the sources and query
    source_embeddings = model.encode(sources)
    query_embedding = model.encode([query])

    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, source_embeddings)[0]

    # Sort results by similarity in descending order
    results = sorted(zip(similarities, sources), reverse=True)
    return results

# Example sources
sources = [
    "How to train a machine learning model",
    "Best practices for deep learning",
    "Introduction to neural networks",
    "Python programming basics",
    "Data preprocessing steps",
    "Best Pizza Tutorial",
    "Machine learning is stealing our jobs",
    "Understanding supervised and unsupervised learning",
    "Artificial intelligence in healthcare",
    "The future of machine learning in business",
    "How to implement neural networks in Python",
    "A guide to reinforcement learning",
    "What is data science and how to get started",
    "Deep learning vs traditional machine learning",
    "Ethical implications of AI technologies",
    "Introduction to convolutional neural networks (CNNs)",
    "How to use natural language processing for text classification",
    "Overview of decision trees and random forests in machine learning",
    "Creating a machine learning model with TensorFlow",
    "The role of AI in self-driving cars",
    "AI in education: Revolutionizing the classroom",
    "The best hiking trails in the world",
    "How to make homemade pizza dough",
    "Meditation techniques for beginners",
    "Top 10 books to read in 2025",
    "The impact of climate change on global ecosystems",
    "Best places to visit in Africa for a cultural experience",
    "Understanding the human brain and how it works",
    "Tips for effective time management in daily life",
    "How to grow organic vegetables in your garden",
    "Yoga poses for stress relief and flexibility",
    "The history of jazz music and its influence on modern genres",
    "The art of painting and how to get started",
    "How to improve your sleep quality",
    "The benefits of a plant-based diet on health",
    "How to organize a successful charity event",
    "Exploring different types of world cuisines",
    "The science behind ocean currents and their effects on marine life"
]



# Example query
query = "Learn about machine learning"

# Perform semantic search
results = semantic_search(sources, query)

# Display results
for similarity, source in results:
    print(f"Similarity: {similarity:.4f} | Source: {source}")
