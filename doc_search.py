from typing import List
from qdrant_client import QdrantClient
from fastembed.embedding import FlagEmbedding as Embedding
import numpy as np
# Example list of documents
documents: List[str] = ['Acute bronchitis', 'Asthma', 'Cancer of the breast', 'Cancer of the colon', 'Cancer of the lung']

# Initialize the DefaultEmbedding class with the desired parameters
embedding_model = Embedding(model_name="BAAI/bge-small-en", max_length=512)

# We'll use the passage_embed method to get the embeddings for the documents
embeddings: List[np.ndarray] = list(
    embedding_model.passage_embed(documents)
)  # notice that we are casting the generator to a list

print(embeddings[0].shape, len(embeddings))
query = "Who was Maharana Pratap?"
query_embedding = list(embedding_model.query_embed(query))[0]
plain_query_embedding = list(embedding_model.embed(query))[0]

def print_top_k(query_embedding, embeddings, documents, k=5):
    scores = np.dot(embeddings, query_embedding)
    sorted_scores = np.argsort(scores)[::-1]
    for i in range(k):
        print(f"Rank {i+1}: {documents[sorted_scores[i]]}")

print_top_k(query_embedding, embeddings, documents)
