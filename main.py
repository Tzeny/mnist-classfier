import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import plot_model
from keras.datasets import mnist

import time
from datetime import datetime


max_epoch=50
optimizer='adam'
dropout_ratio = 0.2
neurons_in_layers = 64
#will be used to give all   
date = time.strftime('%Y-%m-%d-%H:%M')

#load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#dataset:
#   x_train 60000 28x28 uint8_encoded images
#   y_train 60000   [0-9] labels
#   x_test  10000 28x28 uint_8 encoded images
#   y_test  60000   [0-9] labels

#reshape our dataset to make it processable by the NN
x_train = np.reshape(x_train, (60000,784))
x_test = np.reshape(x_test, (10000, 784))

#create a vector with 10 rows for each value in y_train in y_test, to be used by our activation layer
y_train_labels = keras.utils.to_categorical(y_train, num_classes=10)
y_test_labels = keras.utils.to_categorical(y_test, num_classes=10)

model = Sequential([
    Dense(neurons_in_layers, input_shape=(784,)),
    Activation('relu'),
    Dense(neurons_in_layers),
    Activation('relu'),
    Dense(neurons_in_layers),
    Activation('relu'),
    Dense(neurons_in_layers),
    Activation('relu'),
    Dense(neurons_in_layers),
    Activation('relu'),
    Dropout(dropout_ratio),
    Dense(10),
    Activation('softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

plot_model(model, to_file="models-png/" + date + '_model.png',  show_shapes=True)

history = model.fit(x_train, y_train_labels, validation_data=(x_test, y_test_labels), epochs=max_epoch, verbose=1)

#save model
model.save_weights("models/" + optimizer + "_opt-" + str(max_epoch) + "_eps-" + date + ".h5")
print("Saved model to disk")

#plot data

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# plot 3 so that we can better see our model's accuracy
plt.axhline(y=1, color='g', linestyle='--')
plt.axhline(y=0.95, color='orange', linestyle='--')
plt.axhline(y=0.9, color='r', linestyle='--')

plt.title('model - accuracy and loss')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc', 'train_loss', 'val_loss'], loc='upper left')

plt.savefig(optimizer + "_opt-" + str(max_epoch) + "_eps-" + date +'_score.png', bbox_inches='tight')
print("Saved plot to disk")