import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split


train_dataset_file = "datasets/digit-recognizer/train.csv"
test_dataset_file = "datasets/digit-recognizer/test.csv"
train_dataset = pd.read_csv(train_dataset_file)
x_test = pd.read_csv(test_dataset_file)
x_test = x_test.values.astype(np.float32)

x_train, y_train = (train_dataset.iloc[:, 1:].values.astype(np.float32),
                    train_dataset.iloc[:, 0].values.astype(np.float32))
x_train, x_cv, y_train, y_cv = train_test_split(
    x_train, y_train, random_state=42, test_size=0.2
)
x_train, x_cv, x_test = normalize(x_train, axis=1), normalize(x_cv, axis=1), normalize(x_test, axis=1)

x_train = x_train.reshape(-1, 28, 28, 1)
x_cv = x_cv.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss = keras.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=10)

loss, accuracy = model.evaluate(x_train, y_train)
print(f"Training dataset: Loss={loss}, Accuracy={accuracy * 100:.2f}%")
loss, accuracy = model.evaluate(x_cv, y_cv)
print(f"Development dataset: Loss={loss}, Accuracy={accuracy * 100:.2f}%")


result_dataset = model.predict(x_test)
result_labels = np.argmax(result_dataset, axis=1)

result_df = pd.DataFrame({
    "ImageID": np.arange(1, len(result_labels) + 1),
    "Label": result_labels
})

result_df.to_csv("datasets/submission.csv", index=False)
