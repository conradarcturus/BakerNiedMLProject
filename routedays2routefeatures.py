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

# RouteDays data parameters

def main():
    serviceName = "intercitytransit"
    routeName = "route13"

    data = load(serviceName = serviceName, routeName = routeName)

    # Columns
    DISTOLD    = 0
    LAT        = 1
    LON        = 2
    DEV        = 3
    TIMEGLOBAL = 4
    DAYS       = 5
    TIME       = 6
    DIST       = 7
    TRIPID     = 8
    DAYOFWEEK  = 9
    features = ["Old Distance Along Trip (meters)", "Latitude (degrees)", "Longitude (degrees)", "Schedule Deviation (seconds)", "Unix Epoch Time (seconds)", "Days since Schedule Start (days)", "Time in Day (hours)", "Distance Along Trip (meters)", "Trip ID Number", "Day of Week (1 = Monday, ... 5 = Friday)"]
    features_short = ["distold", "lat", "lon", "dev", "timeglobal", "days", "time", "dist", "tripid", "dayofweek"]

    print "Adjusting Features:"

    # Append the day of the week
    data_dayOfWeek = data[:, DAYS].copy() % 7
    data_dayOfWeek.shape = (data.shape[0], 1)
    #print "X_all: {}\tX_dayOfWeek: {}".format(X_all.shape, X_dayOfWeek.shape)
    data = np.append(data, data_dayOfWeek, axis = 1)

    # Normalize the data
    data_norm = np.empty(shape = data.shape)
    for i in range(data.shape[1]):
        data_norm[:,i] = (data[:,i] - data[:,i].mean())/ data[:,i].std()

    print "Saving Feature Files:"

    # Save the files
    path = "/projects/onebusaway/BakerNiedMLProject/data/routefeatures"

    filename = "{}/{}_{}_dist.txt".format(path, serviceName, routeName)
    filedata = data[:, DIST]
    np.savetxt(filename, filedata, fmt="%f")

    filename = "{}/{}_{}_allfeats.txt".format(path, serviceName, routeName)
    filedata = data[:, (DIST, DISTOLD, LAT, LON, TIMEGLOBAL, DAYOFWEEK, DAYS, TIME, TRIPID, DEV)]
    np.savetxt(filename, filedata, fmt="%f")

    filename = "{}/{}_{}_allfeats_normalized.txt".format(path, serviceName, routeName)
    filedata = data_norm[:, (DIST, DISTOLD, LAT, LON, TIMEGLOBAL, DAYOFWEEK, DAYS, TIME, TRIPID, DEV)]
    np.savetxt(filename, filedata, fmt="%f")

    filename = "{}/{}_{}_dist_days_time_dayOfWeek.txt".format(path, serviceName, routeName)
    filedata = data[:, (DIST, DAYS, TIME, DAYOFWEEK)]
    np.savetxt(filename, filedata, fmt="%f")

    filename = "{}/{}_{}_dist_days_time_dayOfWeek_normalized.txt".format(path, serviceName, routeName)
    filedata = data_norm[:, (DIST, DAYS, TIME, DAYOFWEEK)]
    np.savetxt(filename, filedata, fmt="%f")

    filename = "{}/{}_{}_dev.txt".format(path, serviceName, routeName)
    filedata = data[:, DEV]
    np.savetxt(filename, filedata, fmt="%f")

    filename = "{}/{}_{}_timeglobal.txt".format(path, serviceName, routeName)
    filedata = data[:, TIMEGLOBAL]
    np.savetxt(filename, filedata, fmt="%f")

    # Generate Plots
    path = "/projects/onebusaway/BakerNiedMLProject/figures/features"
    p = plot(data_norm, data[:, DEV], '+')
    ylabel(features[DEV])
    xlabel("Normalized Features (mean = 0, stdev = 1)")
    title("{} {}".format(serviceName, routeName))
    legend(p, features_short, loc=4)
    savefig("{}/{}_{}_normfeats_dev.png".format(path, serviceName, routeName))
    
    print "Saving Figures:"
    
    for x in range(10):
        for y in range(10):
            clf()
            xdata = data[:, x]
            ydata = data[:, y]
            if (x == TIME):
                xdata = data[:, x].copy() / 3600
            if (y == TIME):
                ydata = data[:, y].copy() / 3600
            plot(xdata, ydata, '+')
            ylabel(features[y])
            xlabel(features[x])
            title("{} {}".format(serviceName, routeName))
            if (x == DAYOFWEEK):
                xlim(0, 6)
            if (y == DAYOFWEEK):
                ylim(0, 6)
            if (x == DAYS):
                xlim(min(data[:, DAYS]) - 1, max(data[:, DAYS]) + 1)
            if (y == DAYS):
                ylim(min(data[:, DAYS]) - 1, max(data[:, DAYS]) + 1)
            savefig("{}/{}_{}_{}_{}.png".format(path, serviceName, routeName, features_short[x], features_short[y]))
    
    # Clarify time a little better
    clf()
    X_day = data[:, DAYS].copy()
    X_time = data[:, TIME].copy()
    days = unique(X_day)
    N_days = len(days)
    #daysstr = list()
    for i in range(N_days):
        sel_day = data[:, DAYS] == days[i]
        times = data[sel_day, TIME]/3600;
        tsort = times.argsort();
        deviations = data[sel_day, DEV];
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

def load(serviceName, routeName, path = "/projects/onebusaway/BakerNiedMLProject/data/routedays"):
    print "Loading Files:"
    
    # Column Parameters from routedays
    DISTALONGTRIP = 0
    LATITUDE      = 1
    LONGITUDE     = 2
    DEVIATION     = 3
    TIMEGLOBAL    = 4
    TRIPID        = 5
    TIMEALONGTRIP = 6
    DATE          = 7
    TIME          = 8
    DIST          = 9
    
    cols2use = [DISTALONGTRIP, LATITUDE, LONGITUDE, DEVIATION, TIMEGLOBAL, DATE, TIME, DIST, TRIPID]
    
    # Load the files
    text_files = [f for f in os.listdir(path) if (f.endswith('.txt') & f.startswith("{}_{}".format(serviceName, routeName)))]
    # Data has distAlongTrip, lat, lon, delay, time, tripid, timeAlongTrip, date, timeInDay
    data_all = np.empty((0, 9), dtype=np.float);
    for files in text_files:
        print "\t" + files
        filename = path + '/' + files
        
	# X Data
        data_file = np.loadtxt(filename, usecols = cols2use, converters = {7: date2days, 8: time2secs}, delimiter=" ")
        if(data_file[0, 5] >= 42) & (data_file[0, 5] <= 76):
            data_all = np.append(data_all, data_file, axis=0)
            print "\t\tAdded"

    # Sort it by global time
    data_all = data_all[data_all[:, TIMEGLOBAL].argsort(), :]
    
    return data_all
    
if __name__ == '__main__':
    main()
