# manyModelsOptimal.py
#
# Authors: Catherine Baker and Conrad Nied

import numpy as np
import common as cmn
import math
from pylab import *

def main():
    ## Parameters ##
    xSet = "allfeats_normalized" # or manyfeats...
    k = 100
    eta = 0.00001
    N_passes = 50

    # Load the Data
    data = cmn.dataset(xSet = xSet)
    data.load(Nparts = 10, validation = True)#, N_points = 4000)

    # Get xs and sizes
    X  = data.xScope
    Xv = data.xVal
    Xt = data.xTest
    Y  = data.yScope
    Yv = data.yVal
    Yt = data.yTest
    T  = data.tScope
    Tv = data.tVal
    Tt = data.tTest
    (N_train, N_feat) = data.xScope.shape
    N_val             = data.xVal.shape[0]
    N_test            = data.xTest.shape[0]

    # Start a file to record
    filename = "{}/{}_{}_weightvalidation_allfeats_Ntrain{}.txt".format(data.resPath, data.serviceName, data.routeName, N_train)
    f = open(filename, 'w')
    #filename = "{}/{}_{}_weightvalidationSum_Ntrain{}.txt".format(data.resPath, data.serviceName, data.routeName, N_train)
    #f2 = open(filename, 'w')

    # Precompute the future mask
    futureMask = makeFutureMask(T, Tt, futureTime = -30 * 60)

    # Compute the distance matrix (since they all use the same)
    #traintile = np.tile(np.reshape(data.xScope, (N_train, N_feat, 1)), (1, 1, N_test));
    #testtile = np.tile(np.transpose(np.reshape(data.xTest, (N_test, N_feat, 1)), (2, 1, 0)), (N_train, 1, 1));
    #difftile = ((traintile - testtile) ** 2)

    N_iterations = N_val * N_passes
    ws   = np.zeros(shape=(N_iterations, N_feat))
    #ws[0, 0] = 1
    rmse = np.zeros(shape=(N_iterations, 1))
    
    # Preliminary Test
    Ypreds = np.zeros((N_test, 1))
    w = ws[0, :].copy()
    w.shape = (1, N_feat)
    for i in range(N_test):
        #w.shape = (1, N_feat)
        i_point = i % N_test
        Xp = Xt[i_point, :].copy()
        Xp.shape = (1, N_feat)
        Yp = Yt[i_point]
        Tp = Tt[i_point]

        # Limit training data to the past
        Tpast = T < (Tp - (30 * 60))
        Xpast = X[Tpast, :]
        Ypast = Y[Tpast]
        N_past = Xpast.shape[0]

        # Find the distance
        diff = Xpast - np.tile(Xp, (N_past, 1))
        diff *= np.tile(w, (N_past, 1))
        dist = sum(diff ** 2, axis = 1)
        
        # Predict Y using kNN
        if(len(dist) < k):
            Ypred = 0
        else:
            Ypred = np.mean(Ypast[dist.argsort()[:k]])
        Ypreds[i] = Ypred
        Yerror = Yp - Ypred
        #str = "t = {}\tx={:.2f}, {:.2f}, {:.2f}, {:.2f}\ty={:.2f}\typred={:.2f}\n".format(i, Xp[0, 0], Xp[0, 1], Xp[0, 2], Xp[0, 3], Yp, Ypred)
        #print str[:(len(str) - 1)]
        #f.write(str)
        
    rmse = cmn.rmse(Yt, Ypreds)
    str = "eta {}\tpass {}\tw={}\trmse = {:.2f}\n".format(eta, 0, w[0, :], rmse)
    #str = "eta {}\tpass {}\tw={:.2f}, {:.2f}, {:.2f}, {:.2f}\trmse = {:.2f}\n\n".format(eta, 0, w[0, 0], w[0, 1], w[0, 2], w[0, 3], rmse)
    print str[:(len(str) - 1)]
    f.write(str)
    #f2.write(str)

    for i_pass in range(N_passes):

        # Improve the weights
        randOrder = np.random.permutation(range(N_val))
        for i in range(N_val * i_pass, N_val * (i_pass + 1)):
            w = ws[i, :].copy()
            w.shape = (1, N_feat)
            i_point = randOrder[i % N_val]
            Xp = Xv[i_point, :].copy()
            Xp.shape = (1, N_feat)
            Yp = Yv[i_point]
            Tp = Tv[i_point]
    
            # Limit training data to the past
            Tpast = T < (Tp - (30 * 60))
            Xpast = X[Tpast, :]
            Ypast = Y[Tpast]
            N_past = Xpast.shape[0]

            # Find the distance
            diff = Xpast - np.tile(Xp, (N_past, 1))
            diff *= np.tile(w, (N_past, 1))
            dist = sum(diff ** 2, axis = 1)
        
            # Predict Y using kNN
            if(len(dist) < k):
               Ypred = 0
            else:
                Ypred = np.mean(Ypast[dist.argsort()[:k]])
            Yerror = Yp - Ypred
            #str = "i = {}\tw={:.2f}, {:.2f}, {:.2f}, {:.2f}\tx={:.2f}, {:.2f}, {:.2f}, {:.2f}\ty={:.2f}\typred={:.2f}\n".format(i, w[0, 0], w[0, 1], w[0, 2], w[0, 3], Xp[0, 0], Xp[0, 1], Xp[0, 2], Xp[0, 3], Yp, Ypred)
            #print str[:(len(str) - 1)]
            #f.write(str)
            
            # Recalculate weights
            w_delta = eta * (Xp * Yerror + w)
            if i < (N_iterations - 1):
                ws[i + 1] = w - w_delta

        #str = "\nFinished validating\tw={:.2f}, {:.2f}, {:.2f}, {:.2f}\n\n".format(w[0, 0], w[0, 1], w[0, 2], w[0, 3])
        #print str[:(len(str) - 1)]
        #f.write(str)

        # Test
        Ypreds = np.zeros((N_test, 1))
        for i in range(N_test):
            #w.shape = (1, N_feat)
            i_point = i % N_test
            Xp = Xt[i_point, :].copy()
            Xp.shape = (1, N_feat)
            Yp = Yt[i_point]
            Tp = Tt[i_point]

            # Limit training data to the past
            Tpast = T < (Tp - (30 * 60))
            Xpast = X[Tpast, :]
            Ypast = Y[Tpast]
            N_past = Xpast.shape[0]

            # Find the distance
            diff = Xpast - np.tile(Xp, (N_past, 1))
            diff *= np.tile(w, (N_past, 1))
            dist = sum(diff ** 2, axis = 1)
        
            # Predict Y using kNN
	    if(len(dist) < k):
               Ypred = 0
            else:
                Ypred = np.mean(Ypast[dist.argsort()[:k]])
            Ypreds[i] = Ypred
            Yerror = Yp - Ypred
            #str = "t = {}\tx={:.2f}, {:.2f}, {:.2f}, {:.2f}\ty={:.2f}\typred={:.2f}\n".format(i, Xp[0, 0], Xp[0, 1], Xp[0, 2], Xp[0, 3], Yp, Ypred)
            #print str[:(len(str) - 1)]
            #f.write(str)
        
        rmse = cmn.rmse(Yt, Ypreds)
        str = "eta {}\tpass {}\tw={}\trmse = {:.2f}\n".format(eta, i_pass + 1, w[0, :], rmse)
        #wstr = "{:.2f} ".format(w[0, :])
        #str = "eta {}\tpass {}\tw={}\trmse = {:.2f}\n\n".format(eta, i_pass + 1, wstr, rmse)
        print str[:(len(str) - 1)]
        f.write(str)
        #f2.write(str)
    
    f.close()
    #f2.close()

def nonsense(dist, futureMask, Y, k):
    dist = np.sum( * weightstile, axis=1)

    ## Try out some models with rhos and ks
    filename = "{}/{}_{}_tests_Ntrain{}.txt".format(data.resPath, data.serviceName, data.routeName, N_train)
    f = open(filename, 'w')
    str = "Model\t{}\t{}\t{}\n".format("Param", "RunTime", "RMSE")
    print str
    f.write(str)

    # Make a list of parameters to test
    rhos = np.ceil(10 ** (np.arange(12) / 2.0))
    ks   = np.ceil( 2 ** (np.arange(15) / 1.5))

    # Allocate data containers
    N_onTime = 1
    N_kernel = len(rhos)
    N_kNN = len(ks)
    N_attempts = N_onTime + N_kernel + N_kNN
    attempts = range(0, N_attempts)
    rmses = np.zeros(shape=(N_attempts), dtype=np.float)
    times = np.zeros(shape=(N_attempts), dtype=np.float)
    models = [""] * N_attempts

    for i in attempts:
        timer = cmn.timer()
        
        if(i < N_onTime):
            yHat = onTime(dist.shape[1])
            models[i] = "onTime\t"
        elif(i < (N_kernel + N_onTime)):
            rho = rhos[i - N_onTime]
            yHat = kernel(dist, futureMask, Y, rho)
            models[i] = "kernel\t{: 5.0f}".format(rho)
        elif(i < (N_kNN + N_kernel + N_onTime)):
            k = ks[i - N_onTime - N_kernel]
            yHat = kNN(dist, futureMask, Y, k)
            models[i] = "kNN\t{: 5.0f}".format(k)

        times[i] = timer.dur()
        rmses[i] = cmn.rmse(data.yTest, yHat)
        str = "{}\t{:.2f}\t{:.2f}\n".format(models[i], times[i], rmses[i])
        print str
        f.write(str)

    f.close()

def kNN(dist, futureMask, Y, k):
    distc = dist.copy()
    distc[futureMask] = np.nan
    ys = Y[distc.argsort(axis=0)[:k]]
    return np.mean(np.ma.MaskedArray(ys, np.isnan(ys)), axis=0)

def kernel(dist, futureMask, Y, rho):
    [N_train, N_test] = dist.shape
    pi = np.exp(-dist / (rho ** 2))
    pi[futureMask] = 0
    return np.sum(pi * np.tile(np.reshape(Y, (N_train, 1)), (1, N_test)), axis=0) / np.max(np.sum(pi, axis=0), 0)

def onTime(N_test):
    return np.zeros((N_test, 1))

def makeFutureMask(timesA, timesB, futureTime = -3600):
    N_A = len(timesA)
    N_B = len(timesB)
    return (np.tile(np.reshape(timesA, (N_A, 1)), (1, N_B)) - np.tile(np.reshape(timesB, (1, N_B)), (N_A, 1))) > futureTime
    
if __name__ == "__main__":
        main()
