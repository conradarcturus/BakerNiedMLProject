import numpy as np
import scipy as sp
import random as r
import code
import math
import timeit
from pylab import *

class dataset:
    def __init__(self, dataPath = "/projects/onebusaway/BakerNiedMLProject/data/routefeatures", resPath = "/projects/onebusaway/BakerNiedMLProject/data/modelPredictions", figPath = "/projects/onebusaway/BakerNiedMLProject/figures/predictions", serviceName = "intercitytransit", routeName = "route13", xSet = "dist", ySet = "dev"):
        self.dataPath = dataPath
        self.resPath = resPath
        self.figPath = figPath
        self.serviceName = serviceName
        self.routeName = routeName
        self.xSet = xSet
        self.ySet = ySet

    def load(self, N_points = -1, validation = False, timeorganized = False):
        xFull = np.loadtxt("{}/{}_{}_{}.txt".format(self.dataPath, self.serviceName, self.routeName, self.xSet), dtype=np.float)
        yFull = np.loadtxt("{}/{}_{}_{}.txt".format(self.dataPath, self.serviceName, self.routeName, self.ySet), dtype=np.float)
        if N_points == -1:
            N_points = len(xFull)
        self.sel   = np.random.permutation(range(len(xFull)))
        split = N_points/4

        self.xTest  = xFull[self.sel[3*split:4*split]]
        self.yTest  = yFull[self.sel[3*split:4*split]]

        if(validation):
            self.xTrain = xFull[self.sel[       :2*split]]
            self.xVal   = xFull[self.sel[2*split:3*split]]
            self.yTrain = yFull[self.sel[       :2*split]]
            self.yVal   = yFull[self.sel[2*split:3*split]]
        else:
            self.xTrain = xFull[self.sel[       :3*split]]
            self.yTrain = yFull[self.sel[       :3*split]]

    def save(self, data, model = "model"):
        np.savetxt("{}/{}_{}_{}_{}_{}.txt".format(self.resPath, self.serviceName, self.routeName, model, self.xSet, self.ySet), data)

    # Save the yHat vector with its xs and real ys
    def saveYHat(self, data, model = "model"):
        np.savetxt("{}/{}_{}_{}_{}_{}.txt".format(self.resPath, self.serviceName, self.routeName, model, self.xSet, self.ySet), cmb(self.xTest, self.yTest, data))

    # Draws a plot of the data and error
    def visualize(self, yHat, specification="model"):
    
        if(self.xTrain.ndim == 1):
            self.xTrain.shape = (len(self.xTrain), 1)
            self.xTest.shape = (len(self.xTest), 1)

        for i in range(self.xTrain.shape[1]):
            clf()
            plot(self.xTrain[:, i], self.yTrain, 'b+')
            plot(np.vstack((self.xTest[:, i], self.xTest[:, i])), np.vstack((self.yTest.T, yHat.T)), 'r')
            plot(self.xTest[:, i], self.yTest, 'rx')
            savefig("{}/{}_{}_{}_feat{}.png".format(self.figPath, self.serviceName, self.routeName, specification, i))
	    ylabel("Schedule Delay")
	    xlabel("Feature {}".format(i))
	    title("{} {} {}".format(self.serviceName, self.routeName, specification))

#def main():
#    load()
#    yHat = model(xTrain, yTrain, xTest, yTest)
#    save(np.append())
    # Visualize and save the images for the model
#    visualize(xTrain, yTrain, xTest, yTest, yHat, "dist_{}NN".format(k))

def cmb(a, b, c):
    if(a.ndim == 1):
        a.shape = (a.shape[0], 1)
    if(b.ndim == 1):
        b.shape = (a.shape[0], 1)
    if(c.ndim == 1):
        c.shape = (a.shape[0], 1)
    return np.append(a, np.append(b, c, axis = 1), axis = 1)

def rmse(y, yhat):
    yhat = y - yhat;
    count = 0;
    for i in range(0,len(y)):
        count += yhat[i] * yhat[i];
    return (count / len(yhat)) ** 0.5

class timer:
    def __init__(self):
        self.reset()
    def reset(self):
        self.time = timeit.default_timer()
    def dur(self):
        return timeit.default_timer() - self.time
    
if __name__ == "__main__":
        main()
