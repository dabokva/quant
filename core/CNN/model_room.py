import numpy as np
import tensorflow as tf

from core.CNN.CNN import CNN_Predecoder

input_shape = (28, 9, 9, 2)
CNN = CNN_Predecoder()
CNN(tf.zeros(shape=(1, *input_shape), dtype=tf.float32))

print(CNN.summary())