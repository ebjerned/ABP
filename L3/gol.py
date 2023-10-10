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

def perform_tests(board, benchCpu=False, dtypes=[tf.float16]):
    times = []

    
    if benchCpu:
        print("CPU Test")
        tf.config.set_visible_devices([], 'GPU')
        tf.debugging.set_log_device_placement(True)
        boardtf = tf.cast(board, dtype=tf.float32)
        time = timed_life(boardtf)
        print("CPU time: " +str(time) + " s")
        times.append(time)

    else:
        for d in dtypes:
            boardtf = tf.cast(board, dtype=d)
            time = timed_life(boardtf)
            print(str(d) + " took: " + str(time) + " s")
            times.append(time)
            print(times)
    print(dtypes)
    print(times)
    return times
        
def timed_life(board):
    tic = timeit.default_timer()
    boardresult = runlife(board, 1000);
    toc = timeit.default_timer();
    
    result = np.count_nonzero(np.cast[np.int32](boardresult));
    print(result)
    assert(result == 2658)
    return (toc-tic)

@tf.function
def runlife(board, iters):
    board = tf.reshape(board, [1, 2048, 2048, 1])
    filter_kernel= tf.cast(np.array([1,1,1,1,0,1,1,1,1]), dtype=board.dtype)
    filter_kernel = tf.reshape(filter_kernel, [3,3, 1,1])
    for i in range(iters):
        count = tf.nn.conv2d(board,filter_kernel, 1, "SAME")
        # In the case of having 2 neighbours, the cell has to be alive. 
        # All counts of 3 will be alive in next iteration.
        survive = tf.logical_and(tf.equal(board, 1), tf.equal(count,2))
        born = tf.equal(count, 3)
        board = tf.cast(tf.logical_or(survive, born),board.dtype)
        
    board = tf.reshape(board, [2048,2048])
    return board


##times = perform_tests(board,benchCpu=False, dtypes=[tf.float16, tf.float32, tf.bfloat16])

## FOR TESTING CPU PERFOMANCE
times = perform_tests(board,benchCpu=True, dtypes=[tf.float16, tf.float32, tf.bfloat16])
