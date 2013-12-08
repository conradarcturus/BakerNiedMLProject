#!/usr/bin/python
from __future__ import division
import math
import numpy as np
import time
import code
import kNN
import datetime

#infoPath = "/projects/onebusaway/research-dataset/dataset.oba/gtfs"
infoPath = "/projects/onebusaway/gtfs"
period = "2013_06_09"
service = "intercity_transit"
infoPeriod = "2013_06_09-modified" #(KCM 2013_06_08-modified)

dataPath = "/projects/onebusaway/data/block_location_records/puget_sound_prod"
# also "/projects/onebusaway/research-dataset/dataset.oba/realtime/feeds"

def main():
  # Load Trip Data
  trip_txt = "{}/{}/{}/trips.txt".format(infoPath, service, infoPeriod)
  trip_data = np.loadtxt(trip_txt, dtype=str, delimiter=',')
  #trip_data = genfromtxt(trip_txt, delimiter=',',dtype=(int, str, str, str, str, int, int, int, int, int), skip_header=1, converters={3: convert_inoutbound})
  
  # Get specific trip_ids (which is Intercity 13)
  trips_route13  = trip_data[:, 0] == '2';
  trips_weekday  = trip_data[:, 1] == '1';
  trips_outbound = trip_data[:, 3] == 'Outbound';
  trip_ids = trip_data[trips_route13 & trips_weekday & trips_outbound, 10].astype(int)

  # Load route and shape data
  #route_txt = "{}/{}/{}/routes.txt".format(infoPath, service, infoPeriod)
  #route_data = np.loadtxt(trip_txt, dtype=str, delimiter=',')
  shapes_txt = "{}/{}/{}/shapes.txt".format(infoPath, service, infoPeriod)
  shape_data = np.loadtxt(shapes_txt, dtype=str, delimiter=',')
  # shape_id,shape_pt_sequence,shape_dist_traveled,shape_pt_lat,shape_pt_lon

  # Format shape data
  shape_id   = "33" # For route 13 outbound
  
  shape_data = shape_data[shape_data[:, 0] == shape_id, 2:]
  shape_data = shape_data.astype(float)
  shape_data[:, 0] *= 1000.0 # Convert to meters
  interval   = 10
  dist_max   = shape_data[-1, 0]
  N_bins     = int(math.ceil(dist_max / interval))
  shape_int  = np.zeros((N_bins, 3))
  a = 0
  for i in range(N_bins):
    di = i * 10
    while(shape_data[a + 1, 0] - di < 0):
      a += 1
    dia = di - shape_data[a, 0]
    dib = shape_data[a + 1, 0] - di
    shape_int[i, 0] = di
    shape_int[i, 1] = (shape_data[a, 1] * dib + shape_data[a + 1, 1] * dia) / (dia + dib)
    shape_int[i, 2] = (shape_data[a, 2] * dib + shape_data[a + 1, 2] * dia) / (dia + dib)   
  
  # Get database file
  dates = []
  for a in range(112):
    dates.append(datetime.date(2013,6,9)+datetime.timedelta(a))
  for date in dates:
    theDate = str(date)
    print "Loading {}".format(theDate)    
    try:
      filename = "{}/log-{}.gz".format(dataPath, theDate)
      data = np.loadtxt(filename, dtype=(str))
      
      # Parse data
      route_selection = [False] * data.shape[0]
      for trip_id in trip_ids.astype(str):
        route_selection |= (data[:, 18] == trip_id) & (data[:, 8] == "NULL")
      data_route = data[route_selection, :]
      data_basicfeats = data_route[:, (4, 5, 6, 9, 12, 18)].astype(float)
      # distanceAlongTrip, Lat, Lon, scheduleDeviation, time, trip_id
      
      #code.interact(local=locals())
      
      # Sort the data in time
      data_sorted = np.array(sorted(data_basicfeats, cmp=lambda x1, x2: int(x1[5] - x2[5]) if x1[5] != x2[5] else int(x1[4] - x2[4])))
      N = data_sorted.shape[0]
      data_sorted[:, 4] /= 1000 # make time seconds (not milliseconds)

      # Compute Time Strings
      data_timeString = [""] * N
      for i in range(N):
        data_timeString[i] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data_sorted[i, 4]))

      # Compute timeAlongTrip
      data_timeAlongTrip = data_sorted[:, 4].copy() # [0] * N
      for trip_id in trip_ids:
        trip_members = data_sorted[:, 5] == trip_id
        if(trip_members.shape[0] >= 1):
          try:
            data_timeAlongTrip[trip_members] -= data_timeAlongTrip[trip_members].min()
          except:
            print "No trip_id {} on {}".format(trip_id, date)

      # Compute new distances
      #shape_dist = np.sum((np.tile(shape_int[:, 1:].reshape(N_bins, 2, 1), (1, 1, N)) - np.tile(np.transpose(data_sorted[:, 1:2].reshape(N, 2, 1), (2, 1, 0)), (N_bins, 1, 1))) ** 2), axis=1)
      #data_newdist = shape_int[dist.argsort(axis=0)[:, 1]]
      #code.interact(local=locals())
      data_newdist = kNN.manyNearestNeighborsVector(shape_int[:, 1:3], shape_int[:, 0], data_sorted[:, 1:3], 1, weights = np.array((1, 1)))

      # Add in the new features
      data_timeAlongTrip.shape = [N, 1]
      data_out = np.append(data_sorted, data_timeAlongTrip, axis=1)

      data_timeString = np.array(data_timeString)
      data_timeString.shape = [N, 1]
      data_out = np.append(data_out, data_timeString, axis=1)

      data_newdist = np.array(data_newdist)
      data_newdist.shape = [N, 1]
      data_out = np.append(data_out, data_newdist, axis=1)
      #code.interact(local=locals())
      # Save Data
      fileout = "/projects/onebusaway/BakerNiedMLProject/data/routedays/intercitytransit_route13_{}.txt".format(date);
      np.savetxt(fileout, data_out, fmt='%s')
    except Exception as inst:
      print "Something went wrong for the data on {}".format(date)
      print type(inst)     # the exception instance

if __name__ == '__main__':
  main()
