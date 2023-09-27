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
#
#boardtf = tf.cast(board, dtype=tf.float16)
#

def perform_tests(board, benchCpu=False, dtypes=[tf.float16]):
    times = []
    for d in dtypes:
        boardtf = tf.cast(board, dtype=d)
        time = timed_life(boardtf)
        print(str(d) + " took: " + str(time) + " s")
        times.append[time]
        print(times)
    
    if benchCpu:
        tf.config.set_visible_devices([], 'GPU')
        tf.debugging.set_log_device_placement(True)
        boardtf = tf.cast(board, dtype=d)
        time = timed_life(boardtf)
        print("CPU time: " +str(time) + " s")
        times.append(time)
    print(dtypes)
    print(times)
    return times
        
    
    
def timed_life(board):
    tic = timeit.default_timer()
    boardresult = runlife(board, 1000);
    toc = timeit.default_timer();
    
    result = np.count_nonzero(np.cast[np.int32](boardresult));
    assert(result == 2658)
    return (toc-tic)

@tf.function
def runlife(board, iters):
    # Init work
    board = tf.reshape(board, [1, 2048, 2048, 1])
    filter_kernel= tf.cast(np.array([1,1,1,1,0,1,1,1,1]), dtype=board.dtype)
    filter_kernel = tf.reshape(filter_kernel, [3,3, 1,1])
    for i in range(iters):
        # Perform convolution
        count = tf.nn.conv2d(board,filter_kernel, 1, "SAME")
        # In the case of having 2 neighbours, the cell has to be alive. 
        # All counts of 3 will be alive in next iteration.
        survive = tf.logical_and(tf.equal(board, 1), tf.equal(count,2))
        born = tf.equal(count, 3)
        
        # Then, update the board by keeping these tensors
        board = tf.cast(tf.logical_or(survive, born),board.dtype)
        
        print(i)
        
    # Final work
    board = tf.reshape(board, [2048,2048])
    return board


times = perform_tests(board,benchCpu=True, dtypes=[tf.float16, tf.float32, tf.bfloat16, tf.uint8, tf.int32, tf.bool])

#tic = timeit.default_timer()
#boardresult = runlife(boardtf, 1000);
#toc = timeit.default_timer();
#print("Compute time: " + str(toc - tic))
#result = np.cast[np.int32](boardresult);
#print("Cells alive at start: " + str(np.count_nonzero(board)))
#print("Cells alive at end:   " + str(np.count_nonzero(result)))
#print(np.count_nonzero(result))
#plt.figure(figsize=(20,20))
#plt.imshow(result[plotstart:plotend,plotstart:plotend])
#plt.show()
