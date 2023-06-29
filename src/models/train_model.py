import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Data pre-processing

X_train = pd.read_csv("../../dataset/csvTrainImages.csv")
Y_train = pd.read_csv("../../dataset/csvTrainLabel.csv")
X_test = pd.read_csv("../../dataset/csvTestImages.csv")
Y_test = pd.read_csv("../../dataset/csvTestLabel.csv")

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

X_train = X_train.reshape(-1,32,32,1)
X_test  = X_test.reshape(-1,32,32,1)

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

print(f"Train Images: {X_train.shape}")
print(f"Test Images: {X_test.shape}")

# Model creation (VGG-16 architecure)

def vgg16(input_shape, num_classes):
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Initialise model using pre-defined function
model = vgg16((32,32,1), 29)

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, epochs=10)

# Print model summary, evaluate metrics 
# model.summary()
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Accuracy: {accuracy}, Loss: {loss}")