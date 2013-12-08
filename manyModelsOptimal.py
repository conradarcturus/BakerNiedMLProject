# manyModelsOptimal.py
#
# Authors: Catherine Baker and Conrad Nied

import numpy as np
import common as cmn
import math

def main():
    ## Parameters ##
    xSet = "dist_days_time_dayOfWeek_normalized"

    # Load the Data
    data = cmn.dataset(xSet = xSet)
    data.load(Nparts = 10)#, N_points = 12000)

    # Set X, Y, xTest and get sizes
    Y     = data.yScope;

    (N_train, N_feat) = data.xScope.shape
    N_test            = data.xTest.shape[0]

    # Precompute the future mask
    futureMask = makeFutureMask(data.tScope, data.tTest, futureTime = -30 * 60)

    # Compute the distance matrix (since they all use the same)
    weights = np.array([3, 0.5, 2, 1])

    timer = cmn.timer()
    traintile = np.tile(np.reshape(data.xScope, (N_train, N_feat, 1)), (1, 1, N_test));
    testtile = np.tile(np.transpose(np.reshape(data.xTest, (N_test, N_feat, 1)), (2, 1, 0)), (N_train, 1, 1));
    weightstile = np.tile(weights.reshape(1, N_feat, 1), (N_train, 1, N_test))
    print "Dist runtime: {:.2f}".format(timer.dur())

    dist = np.sum(((traintile - testtile) ** 2) * weightstile, axis=1)

    ## Try out some models with rhos and ks
    filename = "{}/{}_{}_tests_Ntrain{}.txt".format(data.resPath, data.serviceName, data.routeName, N_train)
    f = open(filename, 'w')
    str = "Model\t{}\t{}\t{}\n".format("Param", "RunTime", "RMSE")
    print str[:len(str)-1]
    f.write(str)

    # Make a list of parameters to test
    rhos = 2 ** (np.arange(-10, 10) / 2.0)
    ks   = np.ceil( 2 ** (np.arange(18) / 1.5))

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
            models[i] = "kernel\t{: 3.2f}".format(rho)
        elif(i < (N_kNN + N_kernel + N_onTime)):
            k = ks[i - N_onTime - N_kernel]
            yHat = kNN(dist, futureMask, Y, k)
            models[i] = "kNN\t{: 5.0f}".format(k)

        times[i] = timer.dur()
        rmses[i] = cmn.rmse(data.yTest, yHat)
        str = "{}\t{:.2f}\t{:.2f}\n".format(models[i], times[i], rmses[i])
        print str[:len(str)-1]
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
