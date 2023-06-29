import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
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

arabic_characters = ['alef', 'beh', 'teh', 'theh', 'jeem', 'hah', 'khah', 'dal', 'thal',
                    'reh', 'zain', 'seen', 'sheen', 'sad', 'dad', 'tah', 'zah', 'ain',
                    'ghain', 'feh', 'qaf', 'kaf', 'lam', 'meem', 'noon', 'heh', 'waw', 'yeh']

# print(f"Train Images: {X_train.shape}")
# print(f"Test Images: {X_test.shape}")

# Model creation (VGG-16 architecure)

def mod(input_shape, num_classes):
    model = Sequential()

    # Convolutional Layer C1
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))

    # Subsampling Layer S2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolutional Layer C3
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'))

    # Subsampling Layer S4
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten the feature maps
    model.add(Flatten())

    # Fully Connected Layer F5
    model.add(Dense(120, activation='relu'))

    # Output Layer F6
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Initialise model using pre-defined function
model = mod((32,32,1), 29)
# model = vgg16((32,32,1), len(np.unique(Y_train))+1)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, epochs=5)

# Evaluate model metrics 
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Accuracy: {accuracy}, Loss: {loss}")

# Save model 
model.save('1.model')