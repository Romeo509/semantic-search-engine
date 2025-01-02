# Semantic Search Engine

A lightweight **Semantic Search Engine** that leverages Natural Language Processing (NLP) to rank search results based on meaning rather than exact keyword matches. Built with **Sentence Transformers** and **Cosine Similarity**, this tool enables users to retrieve relevant sources from a given dataset by understanding the context of a query.

## Features

- **Semantic Search**: Finds the most contextually relevant results for a query.
- **Cosine Similarity**: Utilizes cosine similarity to compare the query with sources and rank results.
- **Lightweight and Fast**: Based on the `paraphrase-MiniLM-L3-v2` model for efficient performance.
- **Customizable**: Easy to modify and expand for different use cases.

## How It Works

1. **Input Sources**: A list of documents or text snippets (e.g., articles, blog posts, or knowledge base entries).
2. **User Query**: The query that the user wants to search for, which can be a natural language question or statement.
3. **Semantic Search**: The query is transformed into an embedding using the Sentence Transformer model, and the similarity between the query and each source is calculated.
4. **Result Ranking**: The sources are ranked by their similarity to the query in descending order, providing the most relevant sources first.

## Example

### Example Sources:

```python
sources = [
    "How to train a machine learning model",
    "Best practices for deep learning",
    "Introduction to neural networks",
    "Python programming basics",
    "Data preprocessing steps",
    "Best Pizza Tutorial",
    "Machine learning is stealing our jobs"
]

```
### Example query

```python
query = "Learn about machine learning"
```

### Search result

```python
Similarity: 0.8921 | Source: How to train a machine learning model
Similarity: 0.8653 | Source: Best practices for deep learning
Similarity: 0.8372 | Source: Introduction to neural networks
...
```


## Setup Instructions


## Setup Instructions

1. Open this notebook in **Google Colab**.
2. Install the necessary libraries by running the following cells:

   ```python
   # Install Sentence Transformers and scikit-learn for semantic search functionality
   !pip install sentence-transformers scikit-learn
   
   # Install Transformers and PyTorch for model handling and computations
   !pip install transformers torch scikit-learn typing
   ```
### Explanation:
* transformers: The library that provides access to pre-trained models for various NLP tasks, including Sentence Transformers.
* torch: PyTorch, the deep learning framework that powers many models (including Sentence Transformers).
* scikit-learn: For calculating cosine similarity and other machine learning tasks.
* typing: Provides type hinting support for Python (not strictly required, but often useful for clarity).

### Use Cases
Search Engines: Improve search quality by returning results based on meaning.
Recommendation Systems: Suggest related articles, products, or content based on user input.
Knowledge Base: Find relevant articles or FAQs for customer support based on query semantics.
