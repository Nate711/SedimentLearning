import glob
import time
from datetime import datetime
from datetime import timedelta
import cv2
import numpy as np
import pandas as pd
import pytz


def load_station_image_lat_long(
        path='/Users/Nathan/Desktop/Turbidity/SedimentLearning/data_formatter/station_img_coordinates.csv'):
    # TODO: make dictionaries pandas objects for simplicity
    '''
    :param path: Path to csv containing table of the lat long and image x y of each polaris data collection site (called stations)
    :return: Return two dictionaries, the first containing the latlong info and the second the yx info. The dict keys are the station IDs
    '''
    df = pd.read_csv(path)

    latlong_coors = zip(df['Lat'], df['Long'])
    latlong = dict(zip(df['Station'], latlong_coors))

    yx_coors = zip(df['Y'], df['X'])
    yx = dict(zip(df['Station'], yx_coors))

    return latlong, yx


def get_scene_folders(path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/'):
    '''
    :param path: Path to folder containing landsat scenes
    :return: full path to each folder inside the parent folder
    '''
    return glob.glob(path + '*/')  # full path of each folder asdfasdf/asdf/asd/fa/LC1209310/


# Technically all the names of sr, cloud, etc are predetermined by a rule combining the scene name and identifer
# so there's no need to search. Could be much faster
def get_scene_imgs(scene_folder):  # path to this particular scene folder
    '''
    :param scene_folder: The folder of the scene, ie asdf/asdf/asd/fa/LC129310/
    :return: The paths to all GEOTIFs
    '''
    imgs = glob.glob(scene_folder + '*.tif')
    assert len(imgs) > 7
    return imgs


def get_sr_band_imgs(all_imgs):  # all imgs is an array of paths
    '''
    :param all_imgs: list of all tif paths
    :return: path to surface reflectance images
    '''
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
    '''
    :param all_imgs: list of all tif paths
    :return: path to cloud tif
    '''
    cloud = [x for x in all_imgs if 'cloud' in x][0]
    assert cloud is not None
    return cloud


def get_metadata_path(scene_folder):
    '''
    :param scene_folder: scene folder full path
    :return: path to metadata text file
    '''
    mtl = glob.glob(scene_folder + '*MTL.txt')[0]
    return mtl


def get_datetime_from_metadata(metadata_path):
    '''
    :param metadata_path: path to metadata text file
    :return: datetime object of the time the satellite image was taken
    '''
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
    '''
    :param: station_locs_dict: dictionary of station latlong coordinates with station ids as keys
    station_image_coors_dict: dict of station image x and y coordinates with station ids as keys
    :return:
    '''
    data_columns = ['date_time', 'cf_mask_quality', 'cloud', 'station_ID', 'lat', 'long', 'landsat_scene', 'reflec_1',
                    'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_6', 'reflec_7']

    data = {}
    for col in data_columns:
        data[col] = np.array([])  # initialize

    landsat_data_path = '/Users/Nathan/Dropbox/SedimentLearning/data/landsat/'
    folders = get_scene_folders(landsat_data_path)

    count = 0
    total_scenes = len(folders)
    for scene_folder in folders:  # code to execute for each scene
        print "Scene: {}    {} / {}".format(scene_folder, count, total_scenes)
        count = count + 1

        # get names of all images in the scene folder
        imgs = get_scene_imgs(scene_folder)

        # get the paths for the different types of image
        sr_paths = get_sr_band_imgs(imgs)
        cf_path = get_cfmask(imgs)
        cloud_path = get_cloud(imgs)

        # get the name of the scene extracted from cf path
        scene_name = cf_path[cf_path.rfind('/') + 1:cf_path.rfind('_')]

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
        # print col, len(data[col]), len(folders) * len(station_image_coors_dict.keys())
        # print data[col]
        assert len(data[col]) == len(folders) * len(station_image_coors_dict.keys())

    return data


def time_and_write_landsat_data():
    '''
    Parse through every landsat image to extract the reflectances, cloud data, and meta data using the get_data function.
    Then turn this data dictionary into a csv by turning it into a pandas dataframe then to csv.
    The time duration to parse and time to write to csv are printed to console.
    :return:
    '''
    before = time.time()

    # get dictionary of station image yx and latlong
    dict = load_station_image_lat_long()

    # parse through all landsat images
    data = get_data(dict)
    print "Time to parse images and create dictionary: " + str(time.time() - before)

    before = time.time()

    # turn to pandas dataframe
    df = pd.DataFrame.from_dict(data)

    # write to csv
    df.to_csv('/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/landsat_data.csv', mode='wb+', index=False)
    print "Time to write dictionary to file: " + str(time.time() - before)


def convert_polaris_to_UTC():
    '''
    Reads Joe's polaris data csv into Pandas DataFrame and then turns the US pacific times into utc times.
    Then writes to new csv. Prints execution times.
    :return: polaris data dataframe
    '''
    before = time.time()

    # read csv
    polaris = pd.read_csv('/Users/Nathan/Dropbox/SedimentLearning/data/polaris/all_polaris_data.csv', low_memory=False)

    # get rid of the units row
    polaris.drop(polaris.index[0], inplace=True)
    # combine date and time cols
    polaris['date_time_UTC'] = polaris.Date + ' ' + polaris.Time

    # strip spaces off of date_time
    # convert to date_time
    # convert PST to UTC
    polaris['date_time_UTC'] = [datetime.strptime(x.strip(), '%m/%d/%Y %H%M') \
                                    .replace(tzinfo=pytz.timezone('US/Pacific')) for x in polaris.date_time_UTC]

    polaris['date_time_UTC'] = [x.astimezone(pytz.timezone('UTC')) for x in polaris.date_time_UTC]
    polaris.to_csv('/Users/Nathan/Dropbox/SedimentLearning/data/polaris/all_polaris_data_UTC.csv', mode='wb+',
                   index=False)
    print 'Time to execute conversion: ' + str(time.time() - before)
    return polaris


def convert_landsat_to_UTC():
    '''
    Sister function to convert_polaris_to_UTC. Reads the landsat data csv and converts gmt times to utc times.
    Then writes to a new csv and times execution.
    :return: return pandas dataframe
    '''
    before = time.time()
    data = pd.read_csv('/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/landsat_data.csv')
    data['date_time_UTC'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') \
                                 .replace(tzinfo=pytz.timezone('GMT')) for x in data['date_time']]
    data['date_time_UTC'] = [x.astimezone(pytz.timezone('UTC')) for x in data['date_time_UTC']]
    data.drop('date_time', axis=1, inplace=True)
    data.to_csv('/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/landsat_data_UTC.csv')
    print 'Time to execute conversion: ' + str(time.time() - before)
    return data


def read_polaris_to_df(polaris_utc_path='/Users/Nathan/Dropbox/SedimentLearning/data/polaris/all_polaris_data_UTC.csv'):
    '''
    :param path to polaris utc data csv
    :return: Reads the polaris utc csv into a pandas dictionary, converts to dataframe and returns it
    '''
    polaris = pd.read_csv(polaris_utc_path,
                          low_memory=False)
    return polaris


def read_landsat_to_df(landsat_utc_path='/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/landsat_data_UTC.csv'):
    '''
    :param path to landsat_utc data csv
    :return: Dataframe of landsat utc data
    '''
    data = pd.read_csv(landsat_utc_path)
    return data


def create_final_filtered_csv(
        landsat_utc_path='/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/landsat_data_UTC.csv',
        polaris_utc_path='/Users/Nathan/Dropbox/SedimentLearning/data/polaris/all_polaris_data_UTC.csv',
        save_path='/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/filtered.csv',
        filter_hours=8):
    '''
    Go through the landsat and polaris data and match each landsat data point to the closest (time wise) polaris data point.
    This is all done in a pandas dataframe which is then written to a csv and returned.
    :param landsat_utc_path:
    :param polaris_utc_path:
    :param save_path: path to save filtered csv to

    :param filter_hours: only data points taken within this many hours will be accepted

    :return: dataframe of filtered data
    '''

    # get dataframes
    landsat_df = read_landsat_to_df(landsat_utc_path)
    polaris_df = read_polaris_to_df(polaris_utc_path)

    # init filtered dataframe
    filtered_df = landsat_df.copy()
    filtered_df['time_diff'] = np.nan

    # put all the polaris keys into the filtered df
    for key in polaris_df.keys():
        if key not in landsat_df.keys():
            filtered_df[key] = np.nan

    # convert the date_time string representations to actual datetime objects
    # Important: because %z (utc offset string) is not a supported directive on python 2.7 I truncate the utc offset which is 0
    landsat_df['date_time_UTC'] = [datetime.strptime(x[:-6], '%Y-%m-%d %H:%M:%S') for x in landsat_df['date_time_UTC']]
    polaris_df['date_time_UTC'] = [datetime.strptime(x[:-6], '%Y-%m-%d %H:%M:%S') for x in polaris_df['date_time_UTC']]

    # Set the landsat_utc column
    filtered_df['landsat_UTC'] = landsat_df['date_time_UTC']
    filtered_df['polaris_UTC'] = np.nan

    # loop through each polaris station (data collection site)
    for station in landsat_df['station_ID'].unique():

        # create a new dataframe with only data from that location
        landsat_subset_df = landsat_df[landsat_df['station_ID'] == station]
        polaris_subset_df = polaris_df[polaris_df['Station Number'] == station]

        # counts how many data points we've written for this station
        count = 0

        # loop through each data point in the landsat subset
        for idx, date_time in zip(landsat_subset_df.index, landsat_subset_df['date_time_UTC']):
            # print progress
            print '{} / {} for location {}'.format(count, len(landsat_subset_df), station)
            count += 1

            # calculate and copy over time difference
            time_diff = np.abs(date_time - polaris_subset_df['date_time_UTC'])

            # find index of smallest time difference
            index_smallest = np.argmin(time_diff)
            filtered_df.loc[idx, 'time_diff'] = time_diff[index_smallest]

            # copy polaris data over to landsat data
            for key in polaris_df.keys():
                if key == 'date_time_UTC': break
                filtered_df.loc[idx, key] = polaris_subset_df.loc[index_smallest, key]

            # copy polaris date_time
            filtered_df.loc[idx, 'polaris_UTC'] = polaris_subset_df.loc[index_smallest, key]

    # drop the date_time_UTC column b/c it is equiv to landsat_UTC column
    filtered_df.drop('date_time_UTC', axis=1, inplace=True)

    # drop the date time columns from polaris data
    filtered_df.drop('Date', axis=1, inplace=True)
    filtered_df.drop('Time', axis=1, inplace=True)

    # IMPORTANT
    # filter data for time differences < filter_hours parameter
    filtered_df = filtered_df[filtered_df.time_diff < timedelta(hours=filter_hours)]

    # write to csv
    filtered_df.to_csv(save_path, mode='wb+', index=False)

    # return df
    return filtered_df


if __name__ == '__main__':
    data_columns = ['date_time', 'cf_mask_quality', 'cloud', 'station_ID', 'lat', 'long', 'landsat_scene', 'reflec_1',
                    'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_6', 'reflec_7']
    # print read_polaris_to_df()
    # print read_landsat_to_df()

    filtered = create_final_filtered_csv()
    print filtered
