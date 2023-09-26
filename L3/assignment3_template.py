#!/usr/bin/env python3
import matplotlib.pyplot as plt
import lifereader
import numpy as np
import tensorflow as tf
import math
import timeit
import platform

if platform.system() == 'Windows':
    board = lifereader.readlife('C:/Users/Erik Bjerned/Documents/Git_repos/ABP/L3/files/BREEDER3.LIF', 2048)
else:
    board = lifereader.readlife('files/BREEDER3.LIF', 2048)
    

plt.figure(figsize=(20,20))
plotstart=924
plotend=1124
plt.imshow(board[plotstart:plotend,plotstart:plotend])

plt.figure(figsize=(20,20))
plt.imshow(board)
#plt.show()

#tf.config.set_visible_devices([], 'GPU')
#tf.debugging.set_log_device_placement(True)





@tf.function
def runlife(board, iters):
    # Init work
    
    for i in range(iters):
        # In each iteration, compute two bool tensors
        # ’survive’ and ’born’: TODO
        boardtf = tf.cast(board, dtype=tf.float16)

        boardtf = tf.reshape(boardtf, [1, 2048, 2048, 1])
        filter_kernel = tf.cast(np.array([1,1,1,1,0,1,1,1,1]), dtype=tf.float16)
        filter_kernel = tf.reshape(filter_kernel, [3,3, 1,1])
        #filter_kernel = tf.constant(filter_kernel, dtype=tf.float16)
        count = tf.nn.conv2d(boardtf,filter_kernel, 1, "SAME")
        survive = tf.map_fn(fn= lambda x: tf.cast((x==3 or x==2), dtype=tf.float16), elems=count)
        survive = tf.cast(survive, dtype=bool)
        born = tf.map_fn(fn= lambda x: tf.cast((x==3), dtype=tf.float16), elems=count)
        born = tf.cast(born, dtype=bool)
        
        #born=tf.nn.conv2d(boardtf,filter_kernel, 1, "SAME")
        # Then, update the board by keeping these tensors
        boardtf = tf.cast(tf.logical_or(survive, born),dtype=tf.float16)
        print(i)
        
    # Final work
    
    return tf.reshape(boardtf, [2048,2048])


tic = timeit.default_timer()
boardresult = runlife(board, 1000);
toc = timeit.default_timer();
print("Compute time: " + str(toc - tic))
result = np.cast[np.int32](boardresult);
print("Cells alive at start: " + str(np.count_nonzero(board)))
print("Cells alive at end:   " + str(np.count_nonzero(result)))
print(np.count_nonzero(result))
plt.figure(figsize=(20,20))
plt.imshow(result[plotstart:plotend,plotstart:plotend])
