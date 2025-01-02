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
    "Machine learning is stealing our jobs"
]

# Example query
query = "Learn about machine learning"

# Perform semantic search
results = semantic_search(sources, query)

# Display results
for similarity, source in results:
    print(f"Similarity: {similarity:.4f} | Source: {source}")
