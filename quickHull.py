import numpy as np
import matplotlib.pyplot as plt
import math
import time

def calcLine(p1, p2):
    A = -1*(p2[1] - p1[1])
    B = p2[0] - p1[0]
    C = -1*(A*p2[0] + B*p2[1])

    return (A, B, C)

def quickHull_(xs, ys, hull, A, B, C, p1, p2):
    if not len(xs):
        return

    denom = math.sqrt(A*A + B*B)
    dt = np.matrix([xs, ys]).transpose()
    dists = np.absolute(np.dot(dt, np.array([A, B])) + C) / denom

    mi = np.argmax(dists)

    mDistCoord = (xs[mi], ys[mi])
   
    hull.append(mDistCoord)

    A1, B1, C1 = calcLine(p1, mDistCoord)
    A2, B2, C2 = calcLine(p2, mDistCoord)

    tx = []
    ty = []
    bx = []
    by = []

    excludedx = []
    excludedy = []

    denom1 = math.sqrt(A1*A1 + B1*B1)
    denom2 = math.sqrt(A2*A2 + B2*B2)

    betaD = ((p2[1] - mDistCoord[1])*(p1[0] - mDistCoord[0]) + (mDistCoord[0] - p2[0])*(p1[1] - mDistCoord[1]))

    alphaD = ((p2[1] - mDistCoord[1])*(p1[0] - mDistCoord[0]) + (mDistCoord[0] - p2[0])*(p1[1] - mDistCoord[1]))

    for i in range(len(xs)):
        if i is not mi:
            alpha = ((p2[1] - mDistCoord[1])*(xs[i] - mDistCoord[0]) + (mDistCoord[0] - p2[0])*(ys[i] - mDistCoord[1])) / alphaD

            beta = ((mDistCoord[1] - p1[1])*(xs[i] - mDistCoord[0]) + (p1[0] - mDistCoord[0])*(ys[i] - mDistCoord[1])) / betaD

            gamma = 1. - alpha - beta

            if gamma > 0 and beta > 0 and alpha <= 0:
                tx.append(xs[i])
                ty.append(ys[i])

            if gamma > 0 and alpha > 0 and beta <= 0:
                bx.append(xs[i])
                by.append(ys[i])

    quickHull_(tx, ty, hull, A2, B2, C2, p2, mDistCoord)
    quickHull_(bx, by, hull, A1, B1, C1, p1, mDistCoord) 


def quickHull(xs, ys):
    t = time.time()
    hull1 = []
    hull2 = []

    mn = np.argmin(xs)
    mx = np.argmax(xs)

    p1 = (xs[mn], ys[mn])
    p2 = (xs[mx], ys[mx])
    hull1.append(p1)
    hull1.append(p2)

    A, B, C = calcLine(p1, p2)

    tx = []
    ty = []
    bx = []
    by = [] 

    denom = math.sqrt(A*A + B*B)
    for i in range(xs.shape[0]):
        if (i is not mn and i is not mx):
            dist = (A*xs[i] + B*ys[i] + C) / denom
            if dist <= 0:
                bx.append(xs[i])
                by.append(ys[i])
            else:
                tx.append(xs[i])
                ty.append(ys[i])

    quickHull_(tx, ty, hull1, A, B, C, p1, p2)
    quickHull_(bx, by, hull2, A, B, C, p1, p2)
    hull = hull1 + hull2
    
    cx = [x for x,y in hull]
    cy = [y for x,y in hull]
    dt = np.matrix([cx, cy]).transpose()
    print time.time() - t 

    for i in range(len(hull)):
        for j in range(len(hull)):
            if i != j:
                a, b, c = calcLine(hull[i], hull[j])
                de = math.sqrt(a*a + b*b)
                dists = (a*xs + b*ys + c) / de

                sign = 1.0
                for k in range(dists.shape[0]):
                    if dists[k]:
                        sign = dists[k] / abs(dists[k])
                dists = dists * sign
                if np.min(dists) >= 0:
                    plt.plot([hull[i][0], hull[j][0]], [hull[i][1], hull[j][1]], 'b')

    plt.plot(tx, ty, 'ro')
    plt.plot(bx, by, 'ro')
    plt.plot([x for x,y in hull], [y for x,y in hull], 'bo')
    
    plt.show()

def generate(numpoints):
    xs = np.random.normal(5.0, 2000.0, (1, numpoints))
    ys = np.random.normal(15.0, 100.0, (1, numpoints))


    return (xs, ys)

xs, ys = generate(2000)
quickHull(xs[0], ys[0])
