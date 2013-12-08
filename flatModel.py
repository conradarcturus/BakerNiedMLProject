import numpy as np
import common as cmn
import math
import code
from pylab import *

def regress(X, Y, xTest, rho, futureMask = np.empty((1)), weights = np.array([1])):
    N_test = xTest.shape[0]
    return np.zeros((N_test, 1))

def main():
    ## Parameters ##
    data = cmn.dataset(xSet = "dist_days_time_dayOfWeek")
    data.load(Nparts = 10)#N_points = 1000)
    futureMask = makeFutureMask(data.tScope, data.tTest)

    rmse = np.zeros(shape=(len(vals)), dtype=np.float)
    
    timer = cmn.timer()
    yHat = regress(data.xScope, data.yScope, data.xTest, 1, futureMask)
    print "onTime\tRuntime = {:.2f}".format(timer.dur())
    rmse[i] = cmn.rmse(data.yTest, yHat)
    print "\tRMSE = {:.2f}".format(rmse[i])
    data.saveYHat(yHat, model = "onTime".format(k))

    # Visualize and save the images for the model
    data.visualize(yHat, "onTime".format(k))

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
    return (1.0/100.0)*(x[0]-xStar[0])**2 + (1.0)*(x[1]-xStar[1])**2 + (1.0/3600.0)*(x[2]-xStar[2])**2 + (1.0/2.0)*(x[3]-xStar[3])**2

# Returns a binary for whether values are in the future or not
def makeFutureMask(timesA, timesB, futureTime = -3600):
    N_A = len(timesA)
    N_B = len(timesB)
    return (np.tile(np.reshape(timesA, (N_A, 1)), (1, N_B)) - np.tile(np.reshape(timesB, (1, N_B)), (N_A, 1))) > futureTime
    
if __name__ == "__main__":
        main()
