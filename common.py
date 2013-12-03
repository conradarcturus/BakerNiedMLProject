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
        # Get the data from the files
        self.xFull = np.loadtxt("{}/{}_{}_{}.txt".format(self.dataPath, self.serviceName, self.routeName, self.xSet), dtype=np.float)
        self.yFull = np.loadtxt("{}/{}_{}_{}.txt".format(self.dataPath, self.serviceName, self.routeName, self.ySet), dtype=np.float)
        self.times = np.loadtxt("{}/{}_{}_timeglobal.txt".format(self.dataPath, self.serviceName, self.routeName), dtype=np.float)

        # Divide the data into sets for Training, Validation, and Testing
        if N_points == -1:
            self.N = len(self.xFull)
        else:
            self.N = N_points
        self.modelSets = np.random.permutation(range(len(self.xFull))) * 4 / self.N

        self.xTest  = self.xFull[self.modelSets == 3]
        self.yTest  = self.yFull[self.modelSets == 3]
        self.tTest  = self.times[self.modelSets == 3]
        self.xScope = self.xFull[self.modelSets < 3]
        self.yScope = self.yFull[self.modelSets < 3]
        self.tScope = self.times[self.modelSets < 3]

        if(validation):
            self.xTrain = self.xFull[self.modelSets <= 1]
            self.xVal   = self.xFull[self.modelSets == 2]
            self.yTrain = self.yFull[self.modelSets <= 1]
            self.yVal   = self.yFull[self.modelSets == 2]
            self.tTrain = self.times[self.modelSets <= 1]
            self.tVal   = self.times[self.modelSets == 2]
        else:
            self.xTrain = self.xFull[self.modelSets <= 2]
            self.yTrain = self.yFull[self.modelSets <= 2]
            self.tTrain = self.times[self.modelSets <= 2]

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
    yhat.shape = y.shape
    ydiff = y - yhat;
    count = 0;
    for i in range(len(yhat)):
        count += ydiff[i] * ydiff[i];
    return (count / len(ydiff)) ** 0.5

class timer:
    def __init__(self):
        self.reset()
    def reset(self):
        self.time = timeit.default_timer()
    def dur(self):
        return timeit.default_timer() - self.time
    
if __name__ == "__main__":
        main()
