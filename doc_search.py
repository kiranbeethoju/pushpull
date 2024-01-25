from typing import List
from qdrant_client import QdrantClient
from fastembed.embedding import FlagEmbedding as Embedding

# Example list of documents
documents: List[str] = [
    "Maharana Pratap was a Rajput warrior king from Mewar",
    "He fought against the Mughal Empire led by Akbar",
    "The Battle of Haldighati in 1576 was his most famous battle",
    "He refused to submit to Akbar and continued guerrilla warfare",
    "His capital was Chittorgarh, which he lost to the Mughals",
    "He died in 1597 at the age of 57",
    "Maharana Pratap is considered a symbol of Rajput resistance against foreign rule",
    "His legacy is celebrated in Rajasthan through festivals and monuments",
    "He had 11 wives and 17 sons, including Amar Singh I who succeeded him as ruler of Mewar",
    "His life has been depicted in various films, TV shows, and books",
]
# Initialize the DefaultEmbedding class with the desired parameters
embedding_model = Embedding(model_name="BAAI/bge-small-en", max_length=512)

# We'll use the passage_embed method to get the embeddings for the documents
embeddings: List[np.ndarray] = list(
    embedding_model.passage_embed(documents)
)  # notice that we are casting the generator to a list

print(embeddings[0].shape, len(embeddings))

def print_top_k(query_embedding, embeddings, documents, k=5):
    scores = np.dot(embeddings, query_embedding)
    sorted_scores = np.argsort(scores)[::-1]
    for i in range(k):
        print(f"Rank {i+1}: {documents[sorted_scores[i]]}")

print_top_k(query_embedding, embeddings, documents)