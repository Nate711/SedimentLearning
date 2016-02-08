import csv
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
from get_data import get_polaris_list

def get_data_at_station_locations(scene_date_time,sr_paths,cf_mask_path,station_locs,station_image_coors):
    # TODO test everything
    '''
    grab all data from a particular scene, format and return as dictionary
    only works for polaris stations?

    computes n number of data points, where n is equal to the number of stations

    will .keys() return the same order each time?

    :param scene_date_time:
    :param sr_paths: multiple image paths
    :param cf_mask_path: image path
    :param station_locs {id, [lat,long]}
    :param station_image_coors: {id,[y,x]}
    :return:
    '''

    data_columns = ['date_time','cf_mask_quality','station ID', 'lat','long','reflec_1','reflec_2','reflec_3','reflec_4','reflec_5','reflec_6',
                    'reflec_7','reflec_8','reflec_9']

    data = {}
    for col in data_columns:
        col = np.array([]) # initialize

    # write station locations
    for ID in station_locs:
        data['lat'] = np.append(data['lat'],station_locs[ID][0])
    for ID in station_locs:
        data['long'] = np.append(data['long'],station_locs[ID][1])

    time = 0 # TODO get from meta data
    data['date_time'] = np.append(data['date_time'],[time]*len(station_image_coors.keys()))

    # write cloud mask into dictionary
    cf = cv2.imread(cf_mask_path,cv2.IMREAD_ANYDEPTH)
    for ID in station_image_coors.keys():
        cloud = cf[tuple(station_image_coors[ID])] # need the tuple cast if station returns an array
        data['cf_mask_quality'] = np.append(data['cf_mask_quality'],cloud)

    # write reflect masks into dictionary
    for sr_file in sr_paths: # must record which band it is somewhere
        sr = cv2.imread(sr_file,cv2.IMREAD_ANYDEPTH)
        band = int(sr_file[-1]) # get band number
        band_key = 'reflec_{}'.format(band)

        # put the reflectance into the dict, append it to the end
        for ID in station_image_coors.keys():
            reflectance = sr[tuple(station_image_coors[ID])] # raw point value, should average? need the tuple cast if station returns array
            data[band_key] = np.append(data[band_key], reflectance)

    for col in data_columns:
        assert len(data[col]) == len(station_image_coors.keys())

    return data # data is dictionary of date_time,cf_mask quality, station iD, lat, long, reflectances

if __name__ == '__main__':
    print get_polaris_list('/Users/Nathan/Dropbox/SedimentLearning/data/polaris/polaris_locations.csv')