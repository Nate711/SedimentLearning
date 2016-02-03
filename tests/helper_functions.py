import csv
import numpy as np
import test_lin_reg
from os import listdir
from os.path import isfile, join

def get_station_locations(path):
    filenames = [path + f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]
    #print filenames

    X,y= test_lin_reg.get_data(filenames,['insitu_lat','insitu_long'],[])
    #print X
    unique_x = np.vstack({tuple(row) for row in X})

    print unique_x.shape
    print unique_x

    min_lat,max_lat = min(unique_x[:,0]),max(unique_x[:,0])
    min_long,max_long = min(unique_x[:,1]),max(unique_x[:,1])

    print 'Bottom right ' + str((min_lat, max_long))
    print 'Top left ' + str((max_lat, min_long))

def get_scenenames_from_metadata(path):
    filenames = [path + f for f in listdir(path) if isfile(join(path,f)) and f.endswith('.csv')]
    #print filenames
    X,y = test_lin_reg.get_data(filenames,['Landsat Scene Identifier'],[])
    print 'Shape of scene names {}'.format(X.shape)

    with open('/Users/Nathan/Desktop/Turbidity/SedimentLearning/tests/scenes.txt','w+b') as f:
        f.write('\n'.join(X[:,0]))
        print 'Done writing scene names'

if __name__ == '__main__':
    data_path = '/Users/Nathan/Dropbox/SedimentLearning/data/full/'
    get_station_locations(data_path)

    meta_path = '/Users/Nathan/Desktop/Turbidity/SedimentLearning/tests/all_scenes/'
    get_scenenames_from_metadata(meta_path)