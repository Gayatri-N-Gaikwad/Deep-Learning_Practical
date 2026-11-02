
# 1. Import libraries
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# 2. Load dataset
data = pd.read_csv("IMDB_Dataset.csv").dropna()

X = data["review"].astype(str).str.lower()
y = data["sentiment"].map({"positive": 1, "negative": 0})

# 3. Tokenize reviews
sentences = [review.split() for review in X]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.2, random_state=42
)

# 5. Train CBOW Word2Vec
w2v = Word2Vec(X_train, vector_size=100, window=5,
               min_count=1, sg=0, epochs=5)

# 6. Convert reviews into average word vectors
def get_vectors(reviews):
    result = []

    for review in reviews:
        vectors = []

        for word in review:
            if word in w2v.wv:
                vectors.append(w2v.wv[word])

        if len(vectors) == 0:
            result.append(np.zeros(100))
        else:
            result.append(np.mean(vectors, axis=0))

    return np.array(result)

X_train = get_vectors(X_train)
X_test = get_vectors(X_test)

# 7. Build model
model = Sequential([
    Flatten(input_shape=(100,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 8. Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=128,
          validation_split=0.2)

# 9. Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy * 100)
