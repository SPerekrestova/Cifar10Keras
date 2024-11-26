# A convolutional neural network that trains on CIFAR-10 images.

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense
from keras.layers import BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar10

# Downloading and pre-processing dataset
(X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw) = cifar10.load_data()

# Selecting a subset of the actual dataset to save training time
(X_train_raw, Y_train_raw) =  (X_train_raw[0:5000], Y_train_raw[0:5000])
(X_test_raw, Y_test_raw) = (X_test_raw[0:500], Y_test_raw[0:500])

X_train = X_train_raw / 255
X_test_all = X_test_raw / 255
X_validation, X_test = np.split(X_test_all, 2)
Y_train = to_categorical(Y_train_raw)
Y_validation, Y_test = np.split(to_categorical(Y_test_raw), 2)

# Building a convolutional neural network from scratch
model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(1000, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))


# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Training phase for CNN model
history = model.fit(X_train, Y_train,
                    validation_data=(X_validation, Y_validation),
                    epochs=1, batch_size=32)
