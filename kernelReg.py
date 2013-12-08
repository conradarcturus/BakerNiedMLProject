import numpy as np
import common as cmn
import math
import code
from pylab import *

def regress(X, Y, xTest, rho = 20, futureMask = np.empty((1)), weights = np.array([1])):
    # Format the vectors
    if(X.ndim == 1):
	    X.shape = (X.shape[0], 1)
    (N_train, N_feat) = X.shape
    N_test = xTest.shape[0]
    #code.interact(local=locals())
    if(N_feat == 4 and weights.size == 1):
	    weights = np.array([3, 0.5, 2, 1])

    # Compute distance (yes, this is super ugly but 10 to 22 times more efficient during runtime)
    traintile = np.tile(np.reshape(X, (N_train, N_feat, 1)), (1, 1, N_test));
    testtile = np.tile(np.transpose(np.reshape(xTest, (N_test, N_feat, 1)), (2, 1, 0)), (N_train, 1, 1));
    weightstile = np.tile(weights.reshape(1, N_feat, 1), (N_train, 1, N_test))
    dist = np.sum(((traintile - testtile) ** 2) * weightstile, axis=1)
    pi = np.exp(-dist / (rho ** 2))
    
    if futureMask.ndim == 1:
	return np.sum(pi * np.tile(np.reshape(Y, (N_train, 1)), (1, N_test)), axis=0) / np.max(np.sum(pi, axis=0), 0)
    else:
	pi[futureMask] = 0
	return np.sum(pi * np.tile(np.reshape(Y, (N_train, 1)), (1, N_test)), axis=0) / np.max(np.sum(pi, axis=0), 0)

def main():
    ## Parameters ##
    data = cmn.dataset(xSet = "dist_days_time_dayOfWeek")
    data.load(Nparts = 10)#N_points = 1000)
    futureMask = makeFutureMask(data.tScope, data.tTest)

    # Try many values of k
    vals = np.ceil(2 ** np.arange(10))
    rmse = np.zeros(shape=(len(vals)), dtype=np.float)
    for i in range(len(vals)):
        k = vals[i]
        
        timer = cmn.timer()
        yHat = regress(data.xScope, data.yScope, data.xTest, k, futureMask)
        print "rho = {}\tRuntime = {:.2f}".format(k, timer.dur())
        rmse[i] = cmn.rmse(data.yTest, yHat)
        print "\tRMSE = {:.2f}".format(rmse[i])
        data.saveYHat(yHat, model = "kernel_{}rho".format(k))

        # Visualize and save the images for the model
        data.visualize(yHat, "kernel_{}rho".format(k))
    
    # Plot the historical RMSE
    clf()
    plot(vals, rmse)
    xlabel("rho Paramater")
    ylabel("Root Mean Squared Error (seconds)")
    title("Kernel Regression Model, RMSE for different rhos")
    savefig("{}/{}_{}_kernel_rho-rmse.png".format(data.figPath, data.serviceName, data.routeName))

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
