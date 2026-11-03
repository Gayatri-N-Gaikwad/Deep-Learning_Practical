
# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# 2. Load dataset
data = pd.read_csv("NASA TurboFan Engine.csv")

features = ["sensor" + str(i) for i in range(1, 22)]
window = 20

# 3. Select healthy data
healthy = data[data["Fault_Label"] == 0]

# 4. Scale sensor values
scaler = StandardScaler()
scaler.fit(healthy[features])

# 5. Create sequences
def make_windows(df):
    X = []
    for unit, group in df.groupby("unit"):
        values = scaler.transform(group[features])
        for i in range(len(values) - window + 1):
            X.append(values[i:i+window])
    return np.array(X)

X_train = make_windows(healthy)
X_test = make_windows(data)

# 6. Build LSTM Autoencoder
inp = Input(shape=(window, len(features)))
x = LSTM(64)(inp)
x = RepeatVector(window)(x)
x = LSTM(64, return_sequences=True)(x)
out = TimeDistributed(Dense(len(features)))(x)

model = Model(inp, out)
model.compile(optimizer="adam", loss="mse")

# 7. Train model
history = model.fit(
    X_train, X_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1
)

# 8. Calculate threshold
train_pred = model.predict(X_train, verbose=0)
train_error = np.mean((X_train - train_pred)**2, axis=(1, 2))
threshold = np.percentile(train_error, 95)

# 9. Predict anomalies
test_pred = model.predict(X_test, verbose=0)
test_error = np.mean((X_test - test_pred)**2, axis=(1, 2))
y_pred = (test_error > threshold).astype(int)

# 10. Actual labels (last cycle of each window)
y_test = []
for unit, group in data.groupby("unit"):
    labels = group["Fault_Label"].values
    for i in range(len(labels) - window + 1):
        y_test.append(labels[i + window - 1])

y_test = np.array(y_test)

# 11. Evaluate model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# 12. Plot loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
