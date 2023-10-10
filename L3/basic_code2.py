#!/usr/bin/env python3
import tensorflow as tf


x = tf.constant([[1.0], [2.0], [3.0]])
y = tf.constant([[3.2], [4.2]])
w = tf.constant([[0.2, 0.4, 0.3, 0.1], [0.3, 0.1, 0.2, 0.4], [0.2, 0.0, 0.5, 0.3]])
v = tf.constant([[0.3, 0.7], [0.4, 0.6], [0.8, 0.2], [1.1,-0.1]])

with tf.GradientTape(persistent=True) as derivator:
	derivator.watch(w)
	hprime = tf.matmul(w, x, True, False)
	h = tf.math.sigmoid(hprime)
	yprime = tf.matmul(v, h, True, False)
	yh = tf.math.sigmoid(yprime)
	mse = tf.keras.losses.MeanSquaredError()
	Q = mse(y, yh)

dQ_dw = derivator.gradient(Q, w)
print("Derivative of function: " + str(dQ_dw))

del derivator
