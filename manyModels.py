# manyModels.py
#
# Authors: Catherine Baker and Conrad Nied

import numpy as np
import common as cmn
import flatModel as onTime
import kernelReg as kernel
import kNN as kNN
import math
from pylab import *

def main():
    ## Parameters ##
    xSet = "dist_days_time_dayOfWeek"
    data = cmn.dataset(xSet = xSet)
    data.load(Nparts = 10, N_points = 4000)
    futureMask = makeFutureMask(data.tScope, data.tTest, futureTime = -30 * 60)

    print "Model\t{}\t{}\t{}".format("Param", "RunTime", "RMSE")

    # Predict on time
    timer = cmn.timer()
    yHat = onTime.regress(data.xScope, data.yScope, data.xTest, 1, futureMask)
    time_onTime = timer.dur()
    rmse_onTime = cmn.rmse(data.yTest, yHat)
    print "onTime\t\t{:.2f}\t{:.2f}".format(time_onTime, rmse_onTime)
    data.saveYHat(yHat, model = "onTime")

    # Visualize and save the images for the model
    data.visualize(yHat, "onTime")

    ## kNN

    # Try many values of k
    ks = np.ceil(2 ** (np.arange(15) / 1.5))
    rmse_kNN = np.zeros(shape=(len(ks)), dtype=np.float)
    time_kNN = np.zeros(shape=(len(ks)), dtype=np.float)
    for i in range(len(ks)):
        k = ks[i]
        
        timer = cmn.timer()
        yHat = kNN.regress(data.xScope, data.yScope, data.xTest, k, futureMask)
        time_kNN[i] = timer.dur()
        rmse_kNN[i] = cmn.rmse(data.yTest, yHat)
        print "kNN\t{: 5.0f}\t{:.2f}\t{:.2f}".format(k, time_kNN[i], rmse_kNN[i])
        data.saveYHat(yHat, model = "{}NN".format(k))

        # Visualize and save the images for the model
        data.visualize(yHat, "{}NN".format(k))
    
    # Plot the historical RMSE
    clf()
    plot(ks, rmse_kNN)
    xlabel("Number of nearest points, k in kNN")
    ylabel("Root Mean Squared Error (seconds)")
    title("kNN Model, RMSE for different ks")
    savefig("{}/{}_{}_k-rmse.png".format(data.figPath, data.serviceName, data.routeName))

    ## Kernel

    # Try many values of k
    rhos = np.ceil(10 ** (np.arange(12) / 2.0))
    rmse_kernel = np.zeros(shape=(len(rhos)), dtype=np.float)
    time_kernel = np.zeros(shape=(len(rhos)), dtype=np.float)
    for i in range(len(rhos)):
        rho = rhos[i]
        
        timer = cmn.timer()
        yHat = kernel.regress(data.xScope, data.yScope, data.xTest, rho, futureMask)
        time_kernel[i] = timer.dur()
        rmse_kernel[i] = cmn.rmse(data.yTest, yHat)
        print "kernel\t{: 5.0f}\t{:.2f}\t{:.2f}".format(rho, time_kernel[i], rmse_kernel[i])
        data.saveYHat(yHat, model = "kernel_{}rho".format(rho))

        # Visualize and save the images for the model
        data.visualize(yHat, "kernel_{}rho".format(rho))
    
    # Plot the historical RMSE
    clf()
    plot(rhos, rmse_kernel)
    xlabel("rho Paramater")
    ylabel("Root Mean Squared Error (seconds)")
    title("Kernel Regression Model, RMSE for different rhos")
    savefig("{}/{}_{}_kernel_rho-rmse.png".format(data.figPath, data.serviceName, data.routeName))

    #resPath = "/projects/onebusaway/BakerNiedMLProject/data/modelPredictions"
    #serviceName = "intercitytransit", routeName = "route13"
    #np.savetxt("{}/rmses_{}_{}_{}_{}_{}.txt".format(resPath, self.serviceName, self.routeName, "kNN", xSet, self.ySet), cmb(self.xTest, self.yTest, data))

def makeFutureMask(timesA, timesB, futureTime = -3600):
    N_A = len(timesA)
    N_B = len(timesB)
    return (np.tile(np.reshape(timesA, (N_A, 1)), (1, N_B)) - np.tile(np.reshape(timesB, (1, N_B)), (N_A, 1))) > futureTime
    
if __name__ == "__main__":
        main()
