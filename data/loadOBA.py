#!/usr/bin/python
from __future__ import division
import numpy as np
import time
import code
#import matplotlib.pyplot as plt
#import gzip

add_ones = lambda a: hstack((ones((a.shape[0])).reshape(a.shape[0], 1),a))
convert_inoutbound = lambda x: (x == "Outbound") if 1 else 0

def main():
  # Olympic Intercity Transit Data
  trip_txt = "/projects/onebusaway/research-dataset/dataset.oba/gtfs/intercity_transit/2013_06_09-modified/trips.txt"
  #trip_data = genfromtxt(trip_txt, delimiter=',',dtype=int, skip_header=1, converters={3: convert_inoutbound})
  trip_data = np.loadtxt(trip_txt, dtype=str, delimiter=',')
  # King Country Metro
  #trip_txt = "/projects/onebusaway/research-dataset/dataset.oba/gtfs/king_county_metro/2013_06_08-modified/trips.txt";
  #trip_data = genfromtxt(trip_txt, delimiter=',',dtype=(int, str, str, str, str, int, int, int, int, int), skip_header=1, converters={3: convert_inoutbound})

  #routes_txt = "/projects/onebusaway/research-dataset/dataset.oba/gtfs/intercity_transit/2013_06_09-modified/routes.txt";
  #routes_data = genfromtxt(trip_txt, delimiter=',')
  
  # Get specific trip_ids (which is Intercity 13)
  trips_route13  = trip_data[:, 0] == '2';
  trips_weekday  = trip_data[:, 1] == '1';
  trips_outbound = trip_data[:, 3] == 'Outbound';
  trip_ids = trip_data[trips_route13 & trips_weekday & trips_outbound, 10].astype(int)
  
  # Get database file
  folder = "/projects/onebusaway/data/block_location_records/puget_sound_prod"
  #folder = "/projects/onebusaway/research-dataset/dataset.oba/realtime/feeds"
  dates = ["2013-06-10", "2013-06-11", "2013-06-12", "2013-06-13", "2013-06-14", "2013-06-17", "2013-06-18", "2013-06-19", "2013-06-20", "2013-06-21"]
  #date = "2013-06-10"
  for date in dates:
    try:
      filename = "{}/log-{}.gz".format(folder, date)
      
      #f = gzip.open(filename, 'rb')
      #updates = f.read()
      #f.close()
      data = np.loadtxt(filename, dtype=(str))
      
      # Parse data
      route_selection = [False] * data.shape[0]
      for trip_id in trip_ids.astype(str):
        route_selection |= data[:, 18] == trip_id
      data_route = data[route_selection, :]
      data_basicfeats = data_route[:, (4, 5, 6, 9, 12, 18)].astype(float)
      # distanceAlongTrip, Lat, Lon, scheduleDeviation, time, trip_id
      
      #code.interact(local=locals())
      
      # Create features
      #data_basicfeats.view('f64, i64, i64, i64').sort(order=['f3', 'f2'], axis=0)
      data_sorted = np.array(sorted(data_basicfeats, cmp=lambda x1, x2: int(x1[5] - x2[5]) if x1[5] != x2[5] else int(x1[4] - x2[4])))
      #data_basicfeats[data_basicfeats[:,2].argsort(), :]
      N = data_sorted.shape[0]
      data_sorted[:, 4] /= 1000
      data_timeString = [""] * N
      for i in range(N):
        data_timeString[i] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data_sorted[i, 4]))
      data_timeAlongTrip = data_sorted[:, 4] # [0] * N
      for trip_id in trip_ids:
        trip_members = data_sorted[:, 5] == trip_id
        if(trip_members.shape[0] >= 1):
          try:
            data_timeAlongTrip[trip_members] -= data_timeAlongTrip[trip_members].min()
  	  except:
  	    print "No trip_id {} on {}".format(trip_id, date)
      data_timeAlongTrip.shape = [N, 1]
      data_out = np.append(data_sorted, data_timeAlongTrip, axis=1)
      data_timeString = np.array(data_timeString)
      data_timeString.shape = [N, 1]
      data_out = np.append(data_out, data_timeString, axis=1)
      
      # Save Data
      fileout = "/projects/onebusaway/BakerNiedMLProject/intercitytransit_route13_{}.txt".format(date);
      np.savetxt(fileout, data_out, fmt='%s')
    except:
      print "Something went wrong for the data on {}".format(date)

  #gz = tarfile.open(name=filename, mode='r:gz')
  #gz = tarfile.open(name=filename, mode='r:gz', bufsize=10240)
  #all_data = genfromtxt('clickprediction_data/train.txt', delimiter=',')
  #Y = all_data[:,0]
  #X = add_ones(all_data[:, 1:])
  #Xtest = add_ones(genfromtxt('clickprediction_data/test.txt', delimiter=','))
  #Ytest = genfromtxt('clickprediction_data/test_label.txt')
if __name__ == '__main__':
  main()
