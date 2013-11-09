import numpy as np
import os
import datetime
import code

date2datetime = lambda date: datetime.datetime.strptime(date,"%Y-%m-%d").days
date2timedelta = lambda date: datetime.date(date2datetime(date)) - datetime.date(2013, 06, 10);
date2days = lambda date: date2timedelta(date).days
time2datetime = lambda timestr: datetime.datetime.strptime(timestr,'%H:%M:%S')
time2secs = lambda timestr: time2datetimetime2secs(timestr).second + time2datetimetime2secs(timestr).minute * 60 + time2datetimetime2secs(timestr).hour * 3600

def main():
    path = "/projects/onebusaway/BakerNiedMLProject/data/routedays"
    serviceName = "intercitytransit"
    routeName = "route13"
    text_files = [f for f in os.listdir(path) if (f.endswith('.txt') & f.startswith("{}_{}".format(serviceName, routeName)))]
    # Data has distAlongTrip, lat, lon, delay, time, tripid, timeAlongTrip, date, timeInDay
        #alldata = np.array(
    for files in text_files:
        filename = path + '/' + files
        code.interact(local=locals())
        fileXdata = np.loadtxt(filename, usecols = [0, 7, 8], converters = {7: date2days, 8: time2secs}, delimiter=" ")
        
if __name__ == '__main__':
    main()
