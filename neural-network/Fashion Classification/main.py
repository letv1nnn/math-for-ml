import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import normalize

from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = normalize(x_train), normalize(x_test)

fashion_mnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units=128, activation="relu"),
    Dense(units=64, activation="relu"),
    Dense(units=32, activation="relu"),
    Dense(units=10, activation="softmax"),
])

model.compile(optimizer="adam", loss=keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=15)

for i in range(10):
    img = x_test[i].reshape(1, 28, 28)
    prediction = np.argmax(model.predict(img))
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Label: {fashion_mnist_labels[y_test[i]]}, Prediction: {fashion_mnist_labels[prediction]}")
    plt.show()

