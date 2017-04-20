import numpy as np
import matplotlib.pyplot as plt
import math
import time

def Kmeans(k, data):
    oldkv = np.zeros((k, data.shape[1]))
    np.random.shuffle(data)
    kv = np.random.uniform(-100, 100, size=(k, data.shape[1]))
    initkv = np.copy(kv)
    kmaxs = np.zeros((data.shape[0], k))
    kvs = np.zeros(data.shape[0])

    while np.sum((kv - oldkv)):
        print "iterating"
        oldkv = np.copy(kv)

        for j in range(k):
            shifted = data - kv[j,:]
            squared = np.multiply(shifted, shifted)
            summed = np.dot(squared, np.ones((data.shape[1], 1)))
            kmaxs[:,j] = np.squeeze(np.asarray(summed))
        for j in range(data.shape[0]):
            kvs[j] = np.argmin(kmaxs[j,:])

        for i in range(k):
            c = data[np.where(kvs == float(i))[0], :]
            if not c.shape[0]:
                print("randomly reinit")
                kv[i,:] = np.random.uniform(-100, 100, size=(1, 3))
                continue
            kv[i,:] = np.squeeze(np.asarray(np.sum(c, axis=0) / c.shape[0]))

    if data.shape[1] == 2:
        for i in range(k):
            c = data[np.where(kvs == float(i))[0], :]

            xs = np.squeeze(np.asarray(c[:,0]))
            ys = np.squeeze(np.asarray(c[:,1]))
            plt.plot(xs, ys, 'o')
        plt.show()
    if data.shape[1] == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax = fig.add_subplot(221, projection='3d')
        #ax1 = fig.add_subplot(222, projection='3d')
        #ax2 = fig.add_subplot(223, projection='3d')
        for i in range(k):
            c = data[np.where(kvs == float(i))[0], :]

            xs = np.squeeze(np.asarray(c[:,0]))
            ys = np.squeeze(np.asarray(c[:,1]))
            zs = np.squeeze(np.asarray(c[:,2]))
            
            ax.scatter(xs, ys, zs)
            #ax1.scatter(kv[i, 0], kv[i, 1], kv[i, 2], marker='^')
            #ax2.scatter(initkv[i, 0], initkv[i, 1], initkv[i, 2], marker='^')
        plt.show()

def generate(numpoints, ctr, std):
    centers = np.random.uniform(-ctr, ctr, size=2)
    stdevs = np.random.uniform(1, std, size=2)
    xs = np.random.normal(centers[0], stdevs[0], (1, numpoints))
    ys = np.random.normal(centers[1], stdevs[1], (1, numpoints))


    return (xs[0], ys[0])

def generate3(numpoints, ctr, std):
    centers = np.random.uniform(0, ctr, size=3)
    stdevs = np.random.uniform(1, std, size=3)
    xs = np.random.normal(centers[0], stdevs[0], (1, numpoints))
    ys = np.random.normal(centers[1], stdevs[1], (1, numpoints))
    zs = np.random.normal(centers[2], stdevs[2], (1, numpoints))


    return (xs[0], ys[0], zs[0])

"""
xs = np.array([])
ys = np.array([])
for i in range(4):
    xs1, ys1 = generate(100, i*100, 1000)
    xs = np.concatenate((xs, xs1))
    ys = np.concatenate((ys, ys1))

d = np.matrix([xs, ys]).transpose()
"""

xs = np.array([])
ys = np.array([])
zs = np.array([])
for i in range(5):
    xs1, ys1, zs1 = generate3(100, i*20, 20)
    xs = np.concatenate((xs, xs1))
    ys = np.concatenate((ys, ys1))
    zs = np.concatenate((zs, zs1))

d = np.matrix([xs, ys, zs]).transpose()

Kmeans(5, d)

#plt.plot(xs, ys, 'ro')
#plt.plot(xs1, ys1, 'bo')
#plt.show()
