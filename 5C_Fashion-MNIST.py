
# 1. Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 2. Dataset path
data_path = "fashion mnist"

# 3. Load and preprocess images
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation'
)

# 4. Load pretrained VGG16
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# 5. Freeze VGG16 layers
base_model.trainable = False

# 6. Build model using Sequential
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# 7. Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# 9. Evaluate model
loss, accuracy = model.evaluate(val_data)

print("Loss:", loss)
print("Accuracy:", accuracy)

# 10. Plot accuracy
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.show()

# 11. Fine tuning
base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=2
)

# 12. Prediction
images, labels = next(val_data)

prediction = model.predict(images[:1])

class_names = list(train_data.class_indices.keys())

predicted_class = class_names[np.argmax(prediction[0])]

plt.imshow(images[0])
plt.title("Predicted: " + predicted_class)
plt.axis('off')
plt.show()
