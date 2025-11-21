import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf
from keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# feature scaling using normalization
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

# creating a model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units=128, activation="relu"),
    Dense(units=64, activation="relu"),
    Dense(units=10, activation="softmax")
])

model.compile(optimizer="adam",
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=7)

loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy:", accuracy)
print("Loss:", loss)

model.save("digits.keras")

for i in range(5):
    img = x_test[i].reshape(1, 28, 28)
    prediction = model.predict(img)
    print(f"True label: {y_test[i]}")
    print(f"Predicted: {np.argmax(prediction)}")

    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Label: {y_test[i]}, Prediction: {np.argmax(prediction)}")
    plt.show()

'''
for i in range(1, 6):
    img = cv.imread(f"{i}.png")[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"The result is probably: {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
'''
