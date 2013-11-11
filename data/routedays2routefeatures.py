import numpy as np
import os
import datetime
import code
from pylab import *

date2datetime  = lambda date: datetime.datetime.strptime(date,"%Y-%m-%d")
date2timedelta = lambda date: date2datetime(date)-datetime.datetime(2013, 6, 9)
date2days      = lambda date: date2timedelta(date).days
time2datetime  = lambda timestr: datetime.datetime.strptime(timestr,'%H:%M:%S')
time2secs      = lambda timestr: time2datetime(timestr).second + time2datetime(timestr).minute * 60 + time2datetime(timestr).hour * 3600

def main():
    path = "/projects/onebusaway/BakerNiedMLProject/data/routedays"
    serviceName = "intercitytransit"
    routeName = "route13"
    text_files = [f for f in os.listdir(path) if (f.endswith('.txt') & f.startswith("{}_{}".format(serviceName, routeName)))]
    # Data has distAlongTrip, lat, lon, delay, time, tripid, timeAlongTrip, date, timeInDay
    X_all = np.empty((0, 3), dtype=np.float);
    Y_all = np.empty((0), dtype=np.float);
    for files in text_files:
        print files
        filename = path + '/' + files
        #code.interact(local=locals())

	# X Data
        X_file = np.loadtxt(filename, usecols = [0, 7, 8], converters = {7: date2days, 8: time2secs}, delimiter=" ")
        #X_file = np.loadtxt(filename, usecols = [0], delimiter=" ")
        #X_file = np.loadtxt(filename, usecols = [7], delimiter=" ", converters = {7: date2days})
        #X_file = np.loadtxt(filename, usecols = [8], delimiter=" ", dtype=np.str)
	#print "X_all: {}\tX_file: {}".format(X_all.shape, X_file.shape)
	if(X_all.ndim == 1):
            X_all = X_file
        else:
            X_all = np.append(X_all, X_file, axis=0)

	# Y Data
        Y_file = np.loadtxt(filename, usecols = [3], delimiter=" ")
	#print "Y_all: {}\tY_file: {}".format(Y_all.shape, Y_file.shape)
        Y_all = np.append(Y_all, Y_file, axis=0)

    # Append the day of the week
    X_dayOfWeek = X_all[:,1].copy() % 7
    X_dayOfWeek.shape = (X_all.shape[0], 1)
    #print "X_all: {}\tX_dayOfWeek: {}".format(X_all.shape, X_dayOfWeek.shape)
    X_all = np.append(X_all, X_dayOfWeek, axis = 1)

    # Normalize the data
    X_norm = np.empty(shape=X_all.shape)
    for i in range(X_all.shape[1]):
        X_norm[:,i] = (X_all[:,i] - X_all[:,i].mean())/ X_all[:,i].std()

    # Save the files
    path = "/projects/onebusaway/BakerNiedMLProject/data/routefeatures"
    X_fileout = "{}/{}_{}_dist.txt".format(path, serviceName, routeName)
    #np.savetxt(X_fileout, X_all[:,0], fmt="%f")
    X_fileout = "{}/{}_{}_dist_days_time_dayOfWeek.txt".format(path, serviceName, routeName)
    #np.savetxt(X_fileout, X_all, fmt="%f")
    X_norm_fileout = "{}/{}_{}_dist_days_time_dayOfWeek_normalized.txt".format(path, serviceName, routeName)
    #np.savetxt(X_norm_fileout, X_norm, fmt="%f")
    Y_fileout = "{}/{}_{}_dev.txt".format(path, serviceName, routeName)
    #np.savetxt(Y_fileout, Y_all, fmt="%d")

    # Generate Plots
    path = "/projects/onebusaway/BakerNiedMLProject/figures/"
    p = plot(X_norm, Y_all, '+')
    ylabel("Schedule Deviation (seconds)")
    xlabel("Normalized Features (mean = 0, stdev = 1)")
    title("{} {}".format(serviceName, routeName))
    legend(p, ["Distance Along Trip", "Days since Schedule Start", "Time in Day", "Day of Week"], loc=4)
    savefig("{}/{}_{}_normfeats_dev.png".format(path, serviceName, routeName))

    clf()
    plot(X_all[:, 0], Y_all, '+')
    ylabel("Schedule Deviation (seconds)")
    xlabel("Distance Along Trip (meters)")
    title("{} {}".format(serviceName, routeName))
    savefig("{}/{}_{}_dist_dev.png".format(path, serviceName, routeName))

    clf()
    plot(X_all[:, 1], Y_all, '+')
    ylabel("Schedule Deviation (seconds)")
    xlabel("Days since Schedule Start (days)")
    title("{} {}".format(serviceName, routeName))
    xlim(0, 14)
    savefig("{}/{}_{}_days_dev.png".format(path, serviceName, routeName))

    clf()
    plot(X_all[:, 2], Y_all, '+')
    ylabel("Schedule Deviation (seconds)")
    xlabel("Time in Day (seconds)")
    title("{} {}".format(serviceName, routeName))
    savefig("{}/{}_{}_time_dev.png".format(path, serviceName, routeName))

    clf()
    plot(X_all[:, 3], Y_all, '+')
    ylabel("Schedule Deviation (seconds)")
    xlabel("Day of Week (1 = Monday, ... 5 = Friday)")
    xlim(0, 6)
    title("{} {}".format(serviceName, routeName))
    savefig("{}/{}_{}_dayOfWeek_dev.png".format(path, serviceName, routeName))
    
    # Clarify time a little better
    clf()
    X_day = X_all[:, 1].copy()
    X_time = X_all[:, 2].copy()
    days = unique(X_all[:, 1])
    N_days = len(days)
    #daysstr = list()
    for i in range(N_days):
        sel_day = X_all[:, 1] == days[i]
        times = X_all[sel_day, 2]/3600;
        tsort = times.argsort();
        deviations = Y_all[sel_day];
        plot(times[tsort], deviations[tsort])
        #daysstr[i] = "Day {}".format(days[i])
    xlim(8, 9)
    ylabel("Schedule Deviation (seconds)")
    xlabel("Time in Day (hours)")
    legend(p, ["Day 1", "Day 2", "Day 4", "Day 5", "Day 8", "Day 9", "Day 10", "Day 11", "Day 12"], loc=4)
    title("{} {}".format(serviceName, routeName))
    savefig("{}/{}_{}_timefocus_dev.png".format(path, serviceName, routeName))
    
    # Interactive Mode
    #code.interact(local=locals())
    
    
if __name__ == '__main__':
    main()
