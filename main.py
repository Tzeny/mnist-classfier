import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import plot_model
from keras.datasets import mnist

import time
from datetime import datetime

max_epoch=10
date = time.strftime('%Y-%m-%d-%H:%M')

global_scores = []
global_loss = []
global_time = []

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
    Dense(28, input_shape=(784,)),
    Activation('sigmoid'),
    Dense(10),
    Activation('softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

plot_model(model, to_file=date+'_model.png',  show_shapes=True)

highest_score = 0
start = datetime.now()
#train one epoch and evaluate the model
for i in range(0,max_epoch):
    print("Epoch {}/{}".format(i,max_epoch))
    model.fit(x_train, y_train_labels, verbose=1, epochs=1, batch_size=1)
    scores  = model.evaluate(x_test, y_test_labels)

    print(model.metrics_names)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    if scores[1]*100 > highest_score:
        highest_score = scores[1]*100
    global_scores.append(scores[1]*100)
    global_loss.append(scores[0]*100)
    
    delta = datetime.now() - start
    global_time.append(delta.total_seconds()/60)
        
print("Best score: " + str(highest_score))

#save model

model.save_weights("models/" + date + ".h5")
print("Saved model to disk")

#plot data

x = [l for l in range(0,max_epoch)]

line_acc, = plt.plot(global_scores, label='Accuracy (%)')
line_loss, = plt.plot(global_loss, label='Loss')
line_time, = plt.plot(global_time, label='Time (min)')
plt.legend(handles=[line_acc, line_loss, line_time])

#plt.plot(x, global_scores, global_loss)

plt.xlabel('Epoch')
plt.text(0,0,'Max epochs ' + str(max_epoch) + ';' + date, fontsize=6)

plt.savefig(date+'_score.png', bbox_inches='tight')
