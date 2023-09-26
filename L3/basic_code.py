#!/user/bin/env python
import tensorflow as tf
import numpy as np

x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as g:
	g.watch(x)
	y = x*x
	z = y*y

dz_dx = g.gradient(z,y)
dy_dx = g.gradient(y,x)

print("z " + str(z) + " dzdx " + str(dz_dx) + " dydx " + str(dy_dx))

print("z " + str(np.cast[np.float32](z)) + " dzdx " + str(np.cast[np.float32](dz_dx)) + " dydx " + str(np.cast[np.float32](dy_dx)))

del g
