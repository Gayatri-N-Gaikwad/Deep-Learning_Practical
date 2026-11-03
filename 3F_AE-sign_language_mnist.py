
# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# 2. Load CSV dataset
train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")

# 3. Separate features and labels
X = train.drop("label", axis=1).values / 255.0
y = train["label"].values

X_test = test.drop("label", axis=1).values / 255.0
y_test = test["label"].values

# 4. Select only normal classes (exclude label 24)
X_train = X[y != 24]

# 5. Define Autoencoder model
model = Sequential([
    Flatten(input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(784, activation='sigmoid')
])

# 6. Compile model
model.compile(
    optimizer='adam',
    loss='mse'
)

# 7. Train model
history = model.fit(
    X_train, X_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1
)

# 8. Calculate reconstruction error
train_pred = model.predict(X_train, verbose=0)

train_error = np.mean(np.square(X_train - train_pred), axis=1)

# Set threshold
threshold = np.percentile(train_error, 95)

print("Threshold:", threshold)

# 9. Predict anomalies
test_pred = model.predict(X_test, verbose=0)

test_error = np.mean(np.square(X_test - test_pred), axis=1)

y_pred = (test_error > threshold).astype(int)

# Actual labels (Y = 1, remaining = 0)
y_actual = (y_test == 24).astype(int)

# 10. Evaluate model
print("Confusion Matrix:")
print(confusion_matrix(y_actual, y_pred))

print("Classification Report:")
print(classification_report(y_actual, y_pred, zero_division=0))

print("ROC-AUC Score:", roc_auc_score(y_actual, test_error))

# 11. Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Sign Language Autoencoder Loss')
plt.legend()
plt.show()
