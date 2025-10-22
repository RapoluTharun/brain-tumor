import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ------------------------------
# Set dataset path
# ------------------------------
dataset_path = r"C:\Users\LENOVO\OneDrive\Desktop\brain\brain_tumor_dataset"

# ------------------------------
# Data generators
# ------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values
    validation_split=0.2    # 20% data for validation
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# ------------------------------
# Build CNN model
# ------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# ------------------------------
# Compile the model
# ------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ------------------------------
# Train the model
# ------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10  # You can increase to 20-30 if dataset is bigger
)

# ------------------------------
# Save the trained model
# ------------------------------
model.save("brain_tumor_model.h5")

# ------------------------------
# Evaluate model
# ------------------------------
loss, accuracy = model.evaluate(val_gen)
print(f"Validation Accuracy: {accuracy*100:.2f}%")
