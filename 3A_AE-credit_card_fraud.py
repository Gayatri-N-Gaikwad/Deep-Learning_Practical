
# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# 2. Load CSV dataset
data = pd.read_csv("creditcard.csv")

data = data.dropna(subset=["Class"])

# 3. Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.astype(int)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Select only normal transactions
X_train = X_train[y_train == 0]

# 6. Normalize data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Define Autoencoder model
input_dim = X_train.shape[1]

model = Sequential([
    Flatten(input_shape=(input_dim,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(input_dim, activation='linear')
])

# 8. Compile model using Adam
model.compile(
    optimizer='adam',
    loss='mse'
)

# 9. Train model
history = model.fit(
    X_train, X_train,
    epochs=20,
    batch_size=256,
    validation_split=0.1
)

# 10. Calculate reconstruction error
train_pred = model.predict(X_train, verbose=0)

train_error = np.mean(np.square(X_train - train_pred), axis=1)

# Set threshold
threshold = np.percentile(train_error, 95)

print("Threshold:", threshold)

# 11. Predict fraud
test_pred = model.predict(X_test, verbose=0)

test_error = np.mean(np.square(X_test - test_pred), axis=1)

y_pred = (test_error > threshold).astype(int)

# 12. Evaluate model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("ROC-AUC Score:", roc_auc_score(y_test, test_error))

# 13. Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Autoencoder Loss')
plt.legend()
plt.show()
