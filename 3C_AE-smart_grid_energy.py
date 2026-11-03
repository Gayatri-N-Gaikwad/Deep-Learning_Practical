
# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

# 2. Load dataset
data = pd.read_csv("energy_consumption.csv")

data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%d-%m-%Y %H:%M")
data = data.sort_values("Timestamp")

# 3. Separate features and labels
X = data.drop(columns=["Timestamp", "Anomaly_Label"]).values
y = data["Anomaly_Label"].values

# 4. Split dataset (80:20)
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# 5. Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Create time windows
WINDOW = 10

def create_windows(data):
    return np.array([data[i:i+WINDOW] for i in range(len(data)-WINDOW+1)])

X_train = create_windows(X_train)

X_test = create_windows(np.vstack([scaler.transform(
    data.drop(columns=["Timestamp", "Anomaly_Label"]).iloc[split-WINDOW+1:split].values
), X_test]))

y_train = create_windows(y_train)
y_test = create_windows(np.concatenate([y[split-WINDOW+1:split], y[split:]]))

# Select only normal training windows
X_train = X_train[np.all(y_train == 0, axis=1)]

# Mark test windows as normal or anomaly
y_test = np.any(y_test == 1, axis=1).astype(int)

# 7. Define LSTM Autoencoder
model = Sequential([
    LSTM(64, input_shape=(WINDOW, X_train.shape[2])),
    RepeatVector(WINDOW),
    LSTM(64, return_sequences=True),
    TimeDistributed(Dense(X_train.shape[2]))
])

# 8. Compile model
model.compile(optimizer="adam", loss="mse")

# 9. Train model
history = model.fit(
    X_train, X_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

# 10. Calculate reconstruction error
train_pred = model.predict(X_train, verbose=0)
train_error = np.mean(np.square(X_train - train_pred), axis=(1, 2))

# Set threshold
threshold = np.percentile(train_error, 95)
print("Threshold:", threshold)


# 11. Predict anomalies
test_pred = model.predict(X_test, verbose=0)
test_error = np.mean(np.square(X_test - test_pred), axis=(1, 2))
y_pred = (test_error > threshold).astype(int)


# 12. Evaluate model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("ROC_AUC_Score: ", roc_auc_score(y_test, test_error))

# 13. Plot training and validation loss

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Energy Consumption LSTM Autoencoder Loss')
plt.legend()
plt.show()
