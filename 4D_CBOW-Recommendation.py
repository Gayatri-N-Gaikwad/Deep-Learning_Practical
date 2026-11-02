
# 1. Import libraries
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# 2. Read document
with open("document3.txt", "r") as file:
    text = file.read()

# 3. Prepare dataset
text = re.sub(r"[^a-zA-Z\s]", "", text)
words = text.lower().split()

# Divide document into smaller articles
documents = [s.strip().split() for s in text.lower().split(".") if s.strip()]

# 4. Train CBOW model
model = Word2Vec(
    documents, vector_size=50, window=2,
    min_count=1, sg=0, workers=1, epochs=100
)

# 5. Calculate document embeddings
def get_vector(words):
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(50)

doc_vectors = np.array([get_vector(doc) for doc in documents])

# 6. Get user input
query = input("Enter your search text: ")
query = re.sub(r"[^a-zA-Z\s]", "", query).lower().split()

query_vector = get_vector(query)

# 7. Calculate cosine similarity
similarity = cosine_similarity([query_vector], doc_vectors)[0]

# 8. Recommend top 3 documents
top_indices = np.argsort(similarity)[::-1][:3]

print("\nRecommended Documents:")
for i in top_indices:
    print("\n", documents[i])
    print("Similarity:", round(similarity[i], 3))



