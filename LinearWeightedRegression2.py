import numpy as np
import scipy as sp
import random as r
import code
import math
from pylab import *

def linearWeighted(X,Y,xStar,p):

    weight = weightMatrix(xStar,X,p);
    XWeighted = weight*X;
    YWeighted = weight*Y;
    print XWeighted;
    w = sp.stats.linregress(XWeighted,YWeighted);
    return xStar*w[0] + w[1];

def nearestNeighbors(X,Y,xStar,k):
    dist = distMatrix(xStar,X);
    wOrdering = np.argsort(dist);
    sum = 0;
    for i in range(0,k):
        sum += Y[wOrdering[i]];
    return sum/k;

def nearestNeighbors2(X,Y,xStar,k):
    dist = distMatrix2(xStar,X);
    wOrdering = np.argsort(dist);
    sum = 0;
    for i in range(0,k):
        sum += Y[wOrdering[i]];
    return sum/k;


	
    
def weightMatrix(xStar,X,p):
    weight = np.empty([len(X),1]);
    for i in range(0,len(weight)):
        val = math.exp(-dist(X[i],xStar)/(p**2));
        print val;
        weight[i,0] = val;
    print weight;
    return weight;

def distMatrix(xStar,X):
    distance = np.empty([len(X),1]);
    for i in range(0,len(distance)):
        distance[i,0] = dist(X[i],xStar);
    return distance;

def distMatrix2(xStar,X):
    distance = np.empty([len(X),1]);
    for i in range(0,len(distance)):
        distance[i,0] = dist2(X[i],xStar);
    return distance;

# Draws a plot of the data and error
def visualize(xTrain, yTrain, xTest, yTest, yHat, filespec):
    path = "/projects/onebusaway/BakerNiedMLProject/figures"
    serviceName = "intercitytransit"
    routeName = "route13"
    
    if(xTrain.ndim == 1):
        xTrain.shape = (len(xTrain), 1)

    for i in range(xTrain.shape[1]):
        clf()
        plot(xTrain[:, i], yTrain, 'b+')
        plot(xTest[:, i], yTest, 'gx')
        plot(xTest[:, i], yHat, 'rx')
        savefig("{}/{}_{}_{}_feat{}.png".format(path, serviceName, routeName, filespec, i))

def main():
    xFull2 = np.loadtxt("/projects/onebusaway/BakerNiedMLProject/data/routefeatures/intercitytransit_route13_dist_days_time_dayOfWeek_normalized.txt", dtype=np.float)
    yFull2 = np.loadtxt("/projects/onebusaway/BakerNiedMLProject/data/routefeatures/intercitytransit_route13_dev.txt", dtype=np.float)
    N = 400#len(xFull2)
    sel = np.random.permutation(range(N))
    split = N/4
    xTrain2 = xFull2[sel[:split*2]]
    xVal2   = xFull2[sel[split*2:3*split]]
    xTest2  = xFull2[sel[3*split:]]
    yTrain2 = yFull2[sel[:split*2]]
    yVal2   = yFull2[sel[split*2:3*split]]
    yTest2  = yFull2[sel[3*split:]]

    #KNN with more features
    yHat2 = np.zeros(len(yTest2))
    for i in range(0,len(xTest2)):
        if i%100 == 0 and i<>0:
            print str((100.0*i)/len(xTest2)) + "%"
        yHat2[i] = nearestNeighbors2(xTrain2,yTrain2,xTest2[i],100);
    np.savetxt("/projects/onebusaway/BakerNiedMLProject/data/modelPredictions/intercitytransit_route13_dev_predicted100NN_normalized_dist_days_time_dayOfWeek.txt",yHat2)   
    print "rmse = "+str(rmse(yTest2,yHat2))

    visualize(xTrain2, yTrain2, xTest2, yTest2, yHat2, "norm4featmodel")


    #xFull = np.loadtxt("/projects/onebusaway/BakerNiedMLProject/data/routefeatures/intercitytransit_route13_dist.txt", dtype=np.float);
    #yFull = np.loadtxt("/projects/onebusaway/BakerNiedMLProject/data/routefeatures/intercitytransit_route13_dev.txt", dtype=np.float);
    #sel = np.random.permutation(range(len(xFull)));
    #split = len(xFull)/4;
    #xTrain = xFull[sel[:split*2]];
    #xVal = xFull[sel[split*2:3*split]];
    #xTest = xFull[sel[3*split:]];
    #yTrain = yFull[sel[:split*2]];
    #yVal = yFull[sel[split*2:3*split]];
    #yTest = yFull[sel[3*split:]];

    #knn
    
    #yHat = np.zeros(len(yTest));
    #for i in range(0,len(xTest)):
    #if i%100 == 0 and i<>0:
    #        print str((100.0*i)/len(xTest)) + "%";
    #    yHat[i] = nearestNeighbors(xTrain,yTrain,xTest[i],100);
    #np.savetxt("/projects/onebusaway/BakerNiedMLProject/data/modelPredictions/intercitytransit_route13_dev_predicted100NN_dist.txt",yHat);    
    #print "rmse = "+str(rmse(yTest,yHat));
   
    

    #locally weighted linear regression
    #yHat = np.zeros(len(yTest));
    #for i in range(0,len(xTest)):
    #    if i%100 == 0:
    #        print "100 done";
    #    yHat[i] = linearWeighted(xTrain, yTrain, xTest[i],1000);
    #print "rmse = "+str(rmse(yTest,yHat));
    
    


def rmse(y,yhat):
    yhat = y-yhat;
    count = 0;
    for i in range(0,len(y)):
        count += yhat[i]*yhat[i];
    return count/len(yhat);
        
def dist(x, xStar):
    return (x-xStar)**2

def dist2(x, xStar):
    return 3*(x[0]-xStar[0])**2 + .5*(x[1]-xStar[1])**2 + 2*(x[2]-xStar[2])**2 + (x[3]-xStar[3])**2;

def dist3(x, xStar):
    return (1.0/100.0)*(x[0]-xStar[0])**2 + (1.0)*(x[1]-xStar[1])**2 + (1.0/3600.0)*(x[2]-xStar[2])**2 + (1.0/2.0)*(x[3]-xStar[3])**2;
    
if __name__ == "__main__":
        main()
