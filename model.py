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


# X_train_flatten = X_train.reshape(len(X_train), len(X_train[0])**2)
# X_train_flatten.shape
# x_test_flatten = x_test.reshape(len(x_test), len(x_test[0])**2)
# x_test_flatten.shape


# model = Sequential([
#     Flatten(input_shape = (28, 28, )),
#     Dense(100, input_shape = (784,), activation='relu'),
#     Dense(10, activation='softmax')
# ])


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

# prediction_prob = model.predict(x_test)
# prediction = np.array([np.argmax(pred) for pred in prediction_prob])

# print(classification_report(y_test, prediction))

model.save("model.h5")

test = model.predict(x_test[0].reshape(1, 28, 28, 1))
