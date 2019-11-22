# measure-the-over-all-accuracy-of-deep-learning-system
import numpy as np
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf
K.tensorflow_backend._get_available_gpus()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#to measure the overall accuracy of a deep learning system, the number of iterations should be at least 30
NoOfTests = 30
testAccuracy = np.ndarray(shape=(NoOfTests), dtype='float32', order='C')

for k in range(0,NoOfTests):
    K.clear_session()
    network = 0
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    network.fit(train_images, train_labels, epochs=5, batch_size=128, shuffle=True, verbose=0)
    test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=0)
    testAccuracy[k] = test_acc;
    print('The iteration number: ', k, '      The accuracy for this iteration: ', test_acc)

meanTestAccuracy = np.mean(testAccuracy)
print('meanTestAccuracy: ', meanTestAccuracy)


print('finished')

