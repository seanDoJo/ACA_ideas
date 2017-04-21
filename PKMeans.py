from numbapro import cuda, float32, float64
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random
from KMeans import Kmeans

KV = 12
UPDATE_THREADS = 32
UPDATE_BLOCKS = 4

def get_color():
    colors = ['g', 'c', 'y', 'b', 'r', 'm', 'k']
    i = 0
    while True:
        yield colors[i]
        i = (i + 1) % 7

def generate(numpoints, ctr, std):
    centers = np.random.uniform(0, ctr, size=3)
    stdevs = np.random.uniform(1, std, size=3)
    xs = np.random.normal(centers[0], stdevs[0], (1, numpoints))
    ys = np.random.normal(centers[1], stdevs[1], (1, numpoints))
    zs = np.random.normal(centers[2], stdevs[2], (1, numpoints))

    return (xs[0], ys[0], zs[0])

def genCentroids3(k, mxs, mys, mzs, std):
    stdevs = np.random.uniform(0, std, size=3)
    xs = np.random.normal(
	np.random.uniform(mxs[0], mxs[1]),
	stdevs[0], 
	(1, k)
    )
    ys = np.random.normal(
	np.random.uniform(mys[0], mys[1]), 
	stdevs[1], 
	(1, k)
    )

    zs = np.random.normal(
	np.random.uniform(mzs[0], mzs[1]),
	stdevs[2], 
	(1, k)
    )
    ss = np.zeros((1, k))

    d = np.matrix([xs[0], ys[0], zs[0], ss[0]]).T
    flattened = np.squeeze(np.asarray(d.reshape((1, 4*k))))

    return flattened

@cuda.autojit
def closest3(data, assign, centroids, N, k):
    i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    sid = cuda.threadIdx.x
    s_centroids = cuda.shared.array(shape=0, dtype=float64)
    if sid < k*4:
        s_centroids[sid] = centroids[sid]

    if i >= N:
        return
    mDist = -1.0
    ind = -1
    for t in range(k):
        shiftx = data[i*3] - s_centroids[t*4]
        shifty = data[i*3+1] - s_centroids[t*4+1]
        shiftz = data[i*3+2] - s_centroids[t*4+2]
        dist = shiftx*shiftx + shifty*shifty + shiftz*shiftz
        if ind == -1 or dist < mDist:
            mDist = dist
            ind = t
    assign[i] = ind

@cuda.autojit
def update3(data, assign, centroids, N, k):
    index = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    sidx = cuda.threadIdx.x
    nt = cuda.blockDim.x * cuda.gridDim.x

    if index < k*4:
        centroids[index] = 0.

    for i in range(index, N, nt):
        x = data[i*3]
        y = data[i*3 + 1]
        z = data[i*3 + 2]
        a = assign[i]

        cuda.atomic.add(centroids, a*4, x)
        cuda.atomic.add(centroids, a*4 + 1, y)
        cuda.atomic.add(centroids, a*4 + 2, z) 
	cuda.atomic.add(centroids, a*4 + 3, 1)

    if index < k:
    	if centroids[index*4 + 3] >= 1:	
        	centroids[index*4] /= centroids[index*4 + 3]
    		centroids[index*4 + 1] /= centroids[index*4 + 3]
		centroids[index*4 + 2] /= centroids[index*4 + 3]
	else:
		
                n = 3251.
                n = (n*n / 100) % 10000
		centroids[index*4] = n % (101)
		n = (n*n / 100) % 10000
		centroids[index*4 + 1] = n % (101)
		n = (n*n / 100) % 10000
		centroids[index*4 + 2] = n % (101)
		"""
		centroids[index*4] = 10.
		centroids[index*4+1] = 10.
		centroids[index*4+2] = 10.
                """

def PKMeans(dta, k=8, iters=300):	

	N = dta.shape[0]

	KV = k*4

	tpb = 128
	bpg = (N / tpb) + 1

        mnx = dta[np.argmin(dta[:,0]), 0]
	mxx = dta[np.argmax(dta[:,0]), 0]
        mxs = (mnx, mxx)
	mny = dta[np.argmin(dta[:,1]), 1]
	mxy = dta[np.argmax(dta[:,1]), 1]
	mys = (mny, mxy)
	mnz = dta[np.argmin(dta[:,2]), 2]
	mxz = dta[np.argmax(dta[:,2]), 2]
	mzs = (mnz, mxz)

	dh = np.squeeze(np.asarray(dta.reshape((1, dta.shape[0]*dta.shape[1]))))
	ch = genCentroids3(k, mxs, mys, mzs, 10)
	smsize = k*4*ch.dtype.itemsize
	ah = np.zeros(N, dtype=np.int32)

	data = dh.reshape((N, 3))

	stream = cuda.stream()

	dd = cuda.to_device(dh, stream=stream)
	cd = cuda.to_device(ch, stream=stream)
	ad = cuda.to_device(ah, stream=stream)
	
	for i in range(iters):
		closest3[bpg, tpb, stream, smsize](dd, ad, cd, N, k)
		update3[UPDATE_BLOCKS, UPDATE_THREADS, stream](dd, ad, cd, N, k)
	closest3[bpg, tpb, stream, smsize](dd, ad, cd, N, k)

	ad.copy_to_host(ah, stream=stream)
        return ah
	
KK = 7

xs = np.array([])
ys = np.array([])
zs = np.array([])
for i in range(KK):
	xs1, ys1, zs1 = generate(4000, (i+1)*50, 5)
	xs = np.concatenate((xs, xs1))
	ys = np.concatenate((ys, ys1))
	zs = np.concatenate((zs, zs1))

d = np.matrix([xs, ys, zs]).transpose()

start = time.time()
ahh, it = Kmeans(KK, d)
print(time.time() - start)

start = time.time()
ah = PKMeans(d, k=KK, iters=it)
print(time.time() - start)


fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(122, projection='3d')
colors = get_color()

for i in range(KK):
    c = d[np.where(ah == float(i))[0], :]
    col = next(colors)

    xss = np.squeeze(np.asarray(c[:,0]))
    yss = np.squeeze(np.asarray(c[:,1]))
    zss = np.squeeze(np.asarray(c[:,2]))

    ax.scatter(xss, yss, zss, c=col)

for i in range(KK):
    c = d[np.where(ahh == float(i))[0], :]
    col = next(colors)

    xss = np.squeeze(np.asarray(c[:,0]))
    yss = np.squeeze(np.asarray(c[:,1]))
    zss = np.squeeze(np.asarray(c[:,2]))

    ax1.scatter(xss, yss, zss, c=col)
plt.show()
