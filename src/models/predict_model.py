import os
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

arabic_characters = ['alef', 'beh', 'teh', 'theh', 'jeem', 'hah', 'khah', 'dal', 'thal',
                    'reh', 'zain', 'seen', 'sheen', 'sad', 'dad', 'tah', 'zah', 'ain',
                    'ghain', 'feh', 'qaf', 'kaf', 'lam', 'meem', 'noon', 'heh', 'waw', 'yeh']

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

# fig, ax = plt.subplots(3,3, figsize = (30,30))

# for i in range(3):
#     for j in range(3):
#         ax[i,j].imshow(X_train[np.random.randint(0,X_train.shape[0])])
# plt.show()

# Load model 

model = tf.keras.models.load_model("../../models/1.model")
# model.summary()

# cmd = cv2.imread("dad2604.jpg")[:,:,0]
# cmd = np.invert(np.array([cmd]))
# prediction = model.predict(cmd)
# print(f"This character is probably a {np.argmax(prediction)-1}")
# plt.imshow(cmd[0], cmap=plt.cm.binary)
# plt.show()

# img = cv2.imread("hah3258.jpg")[:,:,0]
# img = np.invert(np.array([img]))
# prediction = model.predict(img)
# print(f"This character is probably a {np.argmax(prediction)-1}")
# plt.imshow(img[0], cmap=plt.cm.binary)
# plt.show()

# img = cv2.imread("hah3258.jpg")
# print(img.shape)

# sample = X_test[3358]
# print(sample.shape)

# predictions = model.predict(tf.expand_dims(sample, axis=0))
# predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
# print(predicted_class)
