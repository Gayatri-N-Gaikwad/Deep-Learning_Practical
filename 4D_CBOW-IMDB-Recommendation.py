## # ============================================================
# CONTENT-BASED RECOMMENDATION SYSTEM USING CBOW WORD EMBEDDINGS
# ============================================================

import pandas as pd
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

data = pd.read_csv("twitter_dataset.csv")

print(data.head())
print(data.columns)

# ------------------------------------------------------------
# 2. CLEAN TEXT
# ------------------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

data["clean_text"] = data["Text"].apply(clean_text)

# ------------------------------------------------------------
# 3. TOKENIZE TWEETS
# ------------------------------------------------------------

sentences = data["clean_text"].apply(lambda x: x.split()).tolist()

print("\nFirst Tokenized Tweet:")
print(sentences[0])

# ------------------------------------------------------------
# 4. TRAIN WORD2VEC USING CBOW
# ------------------------------------------------------------

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0
)

print("\nVocabulary Size:", len(model.wv))

# ------------------------------------------------------------
# 5. FUNCTION TO CREATE DOCUMENT VECTOR
# ------------------------------------------------------------

def get_document_vector(words):
    
    vectors = [model.wv[word] for word in words if word in model.wv]
    
    if len(vectors) == 0:
        return np.zeros(100)
    
    return np.mean(vectors, axis=0)

# ------------------------------------------------------------
# 6. CREATE VECTOR FOR EVERY TWEET
# ------------------------------------------------------------

data["vector"] = sentences

document_vectors = np.array([
    get_document_vector(words) for words in sentences
])

print("Document Vectors Shape:", document_vectors.shape)

# ------------------------------------------------------------
# 7. USER INPUT
# ------------------------------------------------------------

user_input = "machine learning technology"

# Clean and tokenize user input
user_input = clean_text(user_input)
user_words = user_input.split()

# Convert user input into vector
user_vector = get_document_vector(user_words)

# ------------------------------------------------------------
# 8. CALCULATE COSINE SIMILARITY
# ------------------------------------------------------------

similarity_scores = cosine_similarity(
    [user_vector],
    document_vectors
)[0]

# ------------------------------------------------------------
# 9. GET TOP 5 RECOMMENDATIONS
# ------------------------------------------------------------

data["similarity"] = similarity_scores

recommendations = data.sort_values(
    by="similarity",
    ascending=False
).head(5)

print("\nTop 5 Recommended Tweets:\n")
print(recommendations)
print(recommendations[["Text", "similarity"]])