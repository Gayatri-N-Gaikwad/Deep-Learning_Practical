# 1. Import libraries
import pandas as pd
import re
from gensim.models import Word2Vec

# 2. Load dataset
data = pd.read_csv("twitter_dataset.csv")
print("Shape of dataset: ", data.shape)
# 3. Clean and tokenize tweets
sentences = []

for tweet in data["Text"].dropna():
    tweet = re.sub(r"http\S+|@\w+|#\w+", "", tweet)
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
    words = tweet.lower().split()

    if len(words) > 0:
        sentences.append(words)

# 4. Train CBOW model
model = Word2Vec(
    sentences,
    vector_size=100,
    window=3,
    min_count=2,
    sg=0,
    workers=4,
    epochs=20
)

# 5. Display word vector
word = "party"

if word in model.wv:
    print("\nWord Vector:")
    print(model.wv[word])

    # 6. Find similar words
    print("\nSimilar Words:")
    print(model.wv.most_similar(word))
else:
    print("Word not found")

# 7. Display vocabulary
print("\nVocabulary:")
print(list(model.wv.index_to_key))