import csv
import numpy as np
import test_lin_reg
from os import listdir
from os.path import isfile, join

mypath = '/Users/Nathan/Dropbox/SedimentLearning/data/full/'

filenames = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.csv')]
print filenames

X,y= test_lin_reg.get_data(filenames,['insitu_lat','insitu_long'],[])
#print X
unique_x = np.vstack({tuple(row) for row in X})

print unique_x.shape
print unique_x

min_lat,max_lat = min(unique_x[:,0]),max(unique_x[:,0])
min_long,max_long = min(unique_x[:,1]),max(unique_x[:,1])

print min_lat, max_long
print max_lat, min_long