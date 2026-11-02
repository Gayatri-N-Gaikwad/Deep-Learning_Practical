
# 1. Import libraries
import re
from gensim.models import Word2Vec

# 2. Read document
with open("document3.txt", "r") as file:
    text = file.read()

# 3. Data preparation
text = re.sub(r"[^a-zA-Z\s]", "", text)
sentences = [text.lower().split()]

# 4. Train CBOW model
model = Word2Vec(
    sentences,
    vector_size=50,
    window=2,
    min_count=1,
    sg=0,
    workers=1,
    epochs=100
)

# 5. Display word embeddings
print("Word Vector:")
print(model.wv["virus"])

# 6. Find similar words
print("\nSimilar words to virus:")
print(model.wv.most_similar("virus", topn=5))

# 7. Check vocabulary
print("\nVocabulary:")
print(model.wv.index_to_key)
