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
    dataPath = "/projects/onebusaway/BakerNiedMLProject/data/routefeatures"
    resPath = "/projects/onebusaway/BakerNiedMLProject/data/modelPredictions"
    figPath = "/projects/onebusaway/BakerNiedMLProject/figures/predictions"
    serviceName = "intercitytransit"
    routeName = "route13"
    xSet = "traj"
    ySet = "dev"
    x = np.loadtxt("{}/{}_{}_{}.txt".format(dataPath, serviceName, routeName, xSet), dtype=np.float)
    

    # Try many values of k
    vals = np.ceil(2 ** np.arange(10))
    rmse = np.zeros(shape=(len(vals)), dtype=np.float)
    minK = 0
    minRMSE = 0
    sel = np.random.permutation(range(len(x)));
    split = len(x)/4;
    xTrain = x[sel[:split*2]];
    xVal = x[sel[split*2:3*split]];
    xTest = x[sel[3*split:]];
    yTest = np.zeros(len(xVal));
    yHat = np.zeros(len(xVal));
    data_norm = np.empty(shape = x.shape)
    theMean = x[:,:].mean()
    theStdDev = x[:,:].std()
    data_norm = (x - theMean)/ theStdDev
    xTrainNorm = data_norm[sel[:split*2]];
    xValNorm = data_norm[sel[split*2:3*split]];
    xTestNorm = data_norm[sel[3*split:]];

    for i in range(len(vals)):
        k = vals[i]
        model = "{}RhoKernelRegress".format(k);
        timer = cmn.timer()
        for j in range(len(xVal)):
            v = len(xVal[0])-15
            t = np.random.randint(10,v);
            yTest[j] = xVal[j][t+10];
            
            #print xTrain[:,:t].shape;
            #print xTrain[:,t+10].shape;
            #print xVal[j,:t].shape;
            #print t;

            yHat[i] = regress(xTrainNorm[:,:t], xTrain[:,t+10], xValNorm[j,:t].reshape(1,t), k, weights=np.ones(t))

        print "rho = {}\tRuntime = {:.2f}".format(k, timer.dur())
        rmse[i] = cmn.rmse(yTest, yHat)
        if i == 0 or rmse[i]<minRMSE:
            minRMSE = rmse[i]
            minK = vals[i]
        print "\tRMSE = {:.2f}".format(rmse[i])
        #data.saveYHat(yHat, model = "kernel_{}rho".format(k))

        # Visualize and save the images for the model
        #data.visualize(yHat, "kernel_{}rho".format(k))
        np.savetxt("{}/{}_{}_{}_{}_val.txt".format(resPath, serviceName, routeName, model, xSet), cmn.cmb(xVal, yTest, yHat))
    k = minK
    yTest = np.zeros(len(xTest));
    yHat = np.zeros(len(xTest));
    model = "{}RhoKernelRegress".format(minK);
    for i in range(len(xTest)):
        v = len(xVal[0])-15
        t = np.random.randint(10,v);
        yTest[i] = xTest[i][t+10];
        yHat[i] = regress(xTrain[:,:t], xTrain[:,t+10], xTest[i,:t].reshape(1,t), k, weights=np.ones(t))
        # Visualize and save the images for the model
        #data.visualize(yHat, "{}NN".format(k))
    np.savetxt("{}/{}_{}_{}_{}_test.txt".format(resPath, serviceName, routeName, model, xSet), cmn.cmb(xTest, yTest, yHat))

    
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
