import numpy as np
import common as cmn
import math
import code
from pylab import *
#import heapq

def nearestNeighbors(X,Y,xStar,k):
    #print "test: x={}".format(xStar)
    dist = distMatrix(xStar,X)
    closest = topK(dist, k)
    return mean(Y[closest])

def manyNearestNeighbors(X, Y, xTest, k):
    N = len(xTest)
    yHat = np.zeros(N)
    for i in range(N):
        #if i%100 == 0 and i<>0:
        #    print str((100.0*i)/N) + "%"
        yHat[i] = nearestNeighbors(X, Y, xTest[i], k)
    return yHat

def manyNearestNeighborsVector(X, Y, xTest, k):
    # Format the vectors
    if(X.ndim == 1):
	X.shape = (X.shape[0], 1)
    (N_train, N_feat) = X.shape
    N_test = xTest.shape[0]
    #code.interact(local=locals())
    weights = np.array([1])
    if(N_feat == 4):
	weights = np.array([3, 0.5, 2, 1])

    # Compute distance (yes, this is super ugly but 10 to 22 times more efficient during runtime)
    dist = np.sum(((np.tile(X.reshape(N_train, N_feat, 1), (1, 1, N_test)) - np.tile(np.transpose(xTest.reshape(N_test, N_feat, 1), (2, 1, 0)), (N_train, 1, 1))) ** 2) * np.tile(weights.reshape(1, N_feat, 1), (N_train, 1, N_test)), axis=1)

    # Return the max values
    return np.mean(Y[dist.argsort(axis=0)[:k]], axis=0)
    

def main():
    ## Parameters ##
    data = cmn.dataset(xSet = "dist_days_time_dayOfWeek")
    data.load()#N_points = 1000)
    k = 100
    timer = cmn.timer()
    yHat = manyNearestNeighborsVector(data.xTrain, data.yTrain, data.xTest, k)
    print "Vectorized Runtime = {:.2f}".format(timer.dur())
    
    print "RMSE = {:.2f}".format(cmn.rmse(data.yTest, yHat))
    data.saveYHat(yHat, model = "{}NN".format(k))

    timer.reset()
    yHat2 = manyNearestNeighbors(data.xTrain, data.yTrain, data.xTest, k)
    print "Iterative Runtime = {:.2f}".format(timer.dur())
    
    print "RMSE = {}".format(cmn.rmse(data.yTest, yHat))

    # Visualize and save the images for the model
    data.visualize(yHat, "{}NN".format(k))

def distMatrix(xStar,X):
    if(len(xStar) == 1):
	distance = np.empty([len(X),1])
        for i in range(len(distance)):
            distance[i,0] = dist(X[i],xStar)
    else:
	distance = np.empty(X.shape)
        for i in range(X.shape[0]):
            distance[i,:] = dist(X[i,:],xStar)
    return distance;

def dist(x, xStar):
    return (x-xStar)**2

def dist2(x, xStar):
    return 3*(x[0]-xStar[0])**2 + .5*(x[1]-xStar[1])**2 + 2*(x[2]-xStar[2])**2 + (x[3]-xStar[3])**2;

def dist3(x, xStar):
    return (1.0/100.0)*(x[0]-xStar[0])**2 + (1.0)*(x[1]-xStar[1])**2 + (1.0/3600.0)*(x[2]-xStar[2])**2 + (1.0/2.0)*(x[3]-xStar[3])**2;

def topK(vals, k):
    return vals.argsort(axis=0)[:k]
    #return -bottleneck.partsort(-vals, k)[:k]
    #return heapq.nsmallest(k, vals)
    #return heapq.nsmallest(k, np.nditer(vals),key=vals.__getitem__)
    
if __name__ == "__main__":
        main()
