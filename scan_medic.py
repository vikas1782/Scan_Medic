#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Set image dimensions
IMG_WIDTH = 128
IMG_HEIGHT = 128

# Paths to training and test (validation) directories
train_data_path = 'path_to_train_data/'  # Update this path to your 'train' folder
test_data_path = 'path_to_test_data/'    # Update this path to your 'test' folder

# Data Augmentation and Image Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=30, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess data
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=32,
    class_mode='categorical'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stop]
)

# Save the model for later use
model.save('tablet_model_classifier.h5')

# Prediction and saving results
output_file_path = "predictions.txt"

# Predicting on each image in the test folder
with open(output_file_path, "w") as file:
    file.write("Filename,Predicted Tablet Model\n")

    for root, _, files in os.walk(test_data_path):
        for img_name in files:
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, img_name)
                img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                img_array /= 255.0  # Normalize the image

                # Make prediction
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)
                
                # Get the predicted class name
                class_names = list(train_generator.class_indices.keys())  # Class names from the training directory
                predicted_class_name = class_names[predicted_class[0]]
                
                # Write prediction to file
                file.write(f"{img_name},{predicted_class_name}\n")

print(f"Predictions saved to {output_file_path}")

