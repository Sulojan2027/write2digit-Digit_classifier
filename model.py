import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.lay  ers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import numpy as np
from keras.layers import Flatten
from sklearn.metrics import classification_report

(X_train, Y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
x_test = x_test / 255
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))



model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train, Y_train, epochs=10)
model.save("model.h5")

test = model.predict(x_test[0].reshape(1, 28, 28, 1))
