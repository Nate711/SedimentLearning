import glob
from datetime import datetime
import cv2
import numpy as np
import csv
import pandas as pd
import time

from get_data import get_polaris_list

def load_station_image_lat_long(path='/Users/Nathan/Desktop/Turbidity/SedimentLearning/data_formatter/station_img_coordinates.csv'):
    df = pd.read_csv(path)

    latlong_coors = zip(df['Lat'],df['Long'])
    latlong = dict(zip(df['Station'],latlong_coors))

    yx_coors = zip(df['Y'],df['X'])
    yx = dict(zip(df['Station'],yx_coors))

    return latlong,yx

def get_scene_folders(path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/'):
    return glob.glob(path + '*/')  # full path of each folder asdfasdf/asdf/asd/fa/LC1209310/


# Technically all the names of sr, cloud, etc are predetermined by a rule combining the scene name and identifer
# so there's no need to search. Could be much faster
def get_scene_imgs(scene_folder):  # path to this particular scene folder
    imgs = glob.glob(scene_folder + '*.tif')
    assert len(imgs) > 7
    return imgs


def get_sr_band_imgs(all_imgs):  # all imgs is an array of paths
    sr = [img for img in all_imgs if 'sr_band' in img]
    assert len(sr) > 5
    return sr


def get_cfmask(all_imgs):
    '''
    :param all_imgs: array of paths of all tifs in folder
    :return: name of cf mask tif
    '''
    # return all_imgs[np.where('cfmask' in all_imgs)][0] # <- alternative
    cf = [x for x in all_imgs if 'cfmask' in x][0]
    assert cf is not None
    return cf


def get_cloud(all_imgs):
    cloud = [x for x in all_imgs if 'cloud' in x][0]
    assert cloud is not None
    return cloud


def get_metadata_path(scene_folder):
    mtl = glob.glob(scene_folder + '*MTL.txt')[0]
    return mtl


def get_datetime_from_metadata(metadata_path):
    date_string = ''
    time_string = ''
    with open(metadata_path, 'rb+') as f:
        # search for date and time tags
        for line in f:
            date_acq = 'DATE_ACQUIRED = '
            index = line.find(date_acq)
            if index is not -1:
                # get the substring from the space after = sign to the end of the line
                date_string = line[len(date_acq) + index:].strip('\n')

            center_time = 'SCENE_CENTER_TIME = '
            index = line.find(center_time)
            if index is not -1:
                # get the substring from the space after = sign to the decimal in the seconds
                time_string = line[len(center_time) + index:line.find('.')].strip('\"\n')

            # both date and time have been found then quit
            if date_string is not '' and time_string is not '':
                break

        # parse the datetime object
        date_time_string = date_string + ' ' + time_string
        dt = datetime.strptime(date_time_string, "%Y-%m-%d %H:%M:%S")
        return dt


def get_data((station_locs_dict, station_image_coors_dict)):
    data_columns = ['date_time', 'cf_mask_quality', 'cloud', 'station_ID', 'lat', 'long', 'landsat_scene', 'reflec_1',
                    'reflec_2', 'reflec_3','reflec_4', 'reflec_5', 'reflec_6','reflec_7']


    data = {}
    for col in data_columns:
        data[col] = np.array([])  # initialize

    landsat_data_path = '/Users/Nathan/Dropbox/SedimentLearning/data/landsat/'
    folders = get_scene_folders(landsat_data_path)

    for scene_folder in folders:  # code to execute for each scene
        # get names of all images in the scene folder
        imgs = get_scene_imgs(scene_folder)

        # get the paths for the different types of image
        sr_paths = get_sr_band_imgs(imgs)
        cf_path = get_cfmask(imgs)
        cloud_path = get_cloud(imgs)

        # get the name of the scene extracted from cf path
        scene_name = cf_path[cf_path.rfind('/')+1:cf_path.rfind('_')]

        data['landsat_scene'] = np.append(data['landsat_scene'], [scene_name] * len(station_image_coors_dict.keys()))
        data['station_ID'] = np.append(data['station_ID'], station_image_coors_dict.keys())

        # write station locations
        for ID in station_locs_dict:
            data['lat'] = np.append(data['lat'], station_locs_dict[ID][0])
        for ID in station_locs_dict:
            data['long'] = np.append(data['long'], station_locs_dict[ID][1])

        # write time
        datetime = str(get_datetime_from_metadata(get_metadata_path(scene_folder)))
        data['date_time'] = np.append(data['date_time'], [datetime] * len(station_image_coors_dict.keys()))

        # write cloud mask into dictionary
        cf_img = cv2.imread(cf_path, cv2.IMREAD_ANYDEPTH)
        for ID in station_image_coors_dict.keys():
            cf = cf_img[tuple(station_image_coors_dict[ID])]  # need the tuple cast if station returns an array
            data['cf_mask_quality'] = np.append(data['cf_mask_quality'], cf)

        # write cloud mask into dictionary
        cloud_img = cv2.imread(cloud_path, cv2.IMREAD_ANYDEPTH)
        for ID in station_image_coors_dict.keys():
            cloud = cloud_img[tuple(station_image_coors_dict[ID])]  # need the tuple cast if station returns an array
            data['cloud'] = np.append(data['cloud'], cloud)

        # initialize list of unwritten bands with all
        unwritten_bands = ['reflec_1', 'reflec_2', 'reflec_3',
                           'reflec_4', 'reflec_5', 'reflec_6',
                           'reflec_7']

        # write reflectance images into dictionary
        for sr_file in sr_paths:  # must record which band it is somewhere
            sr = cv2.imread(sr_file, cv2.IMREAD_ANYDEPTH)
            band = int(sr_file[sr_file.find('.tif') - 1])  # get band number
            band_key = 'reflec_{}'.format(band)

            unwritten_bands.remove(band_key)

            # put the reflectance into the dict, append it to the end
            for ID in station_image_coors_dict.keys():
                # raw point value, should average? need the tuple cast if station returns array
                reflectance = sr[tuple(station_image_coors_dict[ID])]
                data[band_key] = np.append(data[band_key], reflectance)

        # put np.nan
        for band_key in unwritten_bands:
            for ID in station_image_coors_dict.keys():
                reflectance = np.nan  # ERROR VALUE
                data[band_key] = np.append(data[band_key], reflectance)

    # check that there are as many data points as stations
    for col in data_columns:
        #print col, len(data[col]), len(folders) * len(station_image_coors_dict.keys())
        #print data[col]
        assert len(data[col]) == len(folders) * len(station_image_coors_dict.keys())

    return data

if __name__ == '__main__':

    print get_polaris_list('/Users/Nathan/Dropbox/SedimentLearning/data/polaris/polaris_locations.csv')
    before = time.time()
    dict = load_station_image_lat_long()
    data =  get_data(dict)
    print "Time to parse images and create dictionary: " + str(time.time()-before)

    data_columns = ['date_time', 'cf_mask_quality', 'cloud', 'station_ID', 'lat', 'long', 'landsat_scene', 'reflec_1',
                    'reflec_2', 'reflec_3','reflec_4', 'reflec_5', 'reflec_6','reflec_7']
    before = time.time()
    df = pd.DataFrame.from_dict(data)

    df.to_csv('/Users/Nathan/Desktop/landsat_data.csv',mode='wb+',index=False)
    print "Time to write dictionary to file: " + str(time.time()-before)
