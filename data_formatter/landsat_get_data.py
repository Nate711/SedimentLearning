import glob
import time
from datetime import datetime
from datetime import timedelta
import cv2
import numpy as np
import pandas as pd
import pytz
import os
from xml.dom import minidom


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


def get_image_xy_in_scene_for_lat_long(path = '/Users/Nathan/Desktop/Turbidity/SedimentLearning/data_formatter/usgs_moorings_locations.csv'):
    '''
    Find the pixel coordinates of the usgs moorings in the landsat images. Print the pixel coordinates
    :return: nothing

    '''

    df = pd.read_csv(path)
    coors = zip(df['Station'],df['Long'], df['Lat'])
    for station,long,lat in coors:
        result = os.popen(
            'gdallocationinfo /Users/Nathan/Dropbox/SedimentLearning/data/landsat/LC80440342013106-SC20160218112047/LC80440342013106LGN01_sr_band1.tif -xml -wgs84 %s %s' % (long,lat)).read()
        xmldoc = minidom.parseString(result)
        itemlist = xmldoc.getElementsByTagName('Report')
        print 'Station: {}   X: {}   Y: {}'.format(station,itemlist[0].attributes['pixel'].value, itemlist[0].attributes['line'].value)


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

def get_cloud_adjacent(all_imgs):
    for x in all_imgs:
        if 'cloud' in x and 'adjacent' in x:
            print x
            return x
    return None


def get_cloud(all_imgs):
    '''
    :param all_imgs: list of all tif paths
    :return: path to cloud tif
    '''
    cloud = [x for x in all_imgs if ('cloud' in x and not 'adjacent' in x)][0]
    print cloud
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

def get_landsat_spacecraft_ID(scene_folder):
    '''
    Get the spacecraft id (ie Landsat 8) from the name of the scene
    For example: /Users/Nathan/Dropbox/SedimentLearning/data/landsat/LC80440342014077-SC20160218112617/
    indicates that the spacecraft is landsat_8 from the LC8 identifier
    :param scene_folder: full path to scene folder
    :return: name, ie "LANDSAT_8"
    '''

    # just get the scene name by searching for the 2nd / from the right
    scene_name = scene_folder[scene_folder[:-1].rfind("/")+1:]
    number = scene_name[2]
    return 'LANDSAT_{}'.format(number)

def remove_data_where_cloud_cover_df(df):
    return df[df.cloud==0]

def remove_data_where_cloud_cover_csv(filtered_csv_path = '/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_*'):
    paths = glob.glob(filtered_csv_path)

    for path in paths:
        df = pd.read_csv(path)

        df = remove_data_where_cloud_cover_df(df)
        df.to_csv(path,index=False)

def convert_landsat8_band_to_landsat457_band(band):
    '''
    landsat 8 band | wavelength | landsat 4,5,7 band
    1 | 430-450 | none
    2 | 450-510 blue  | 1
    3 | 530-590 green | 2
    4 | 640-670 red   | 3
    5 | 805-880 NIR   | 4
    6 | 1570-1650 SWIR| 5
    7 | 2110 - 2290 SWIR 2 | 7
    8 | 500-680 panchromatic | 8
    none | IR | 6
    :param band: landsat 8 band
    :return: landsat 4,5,7 band
    '''
    mapping = {1:None,2:1,3:2,4:3,5:4,6:5,7:7}
    return mapping[band]

def get_scene_name(img_path):
    return img_path[img_path.rfind('/') + 1:img_path.rfind('/') + 22]

def get_landsat_data((station_locs_dict, station_image_coors_dict)):
    # TODO fix description
    # TODO change data data structure from dictionary to pandas dataframe
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
        cloud_adj_path = get_cloud_adjacent(imgs)

        #print scene_folder
        spacecraft_ID = get_landsat_spacecraft_ID(scene_folder)

        # get the name of the scene extracted from cf path
        scene_name = get_scene_name(cf_path)

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
        # count adjacent clouds as cloud
        if cloud_adj_path is not None:
            cloud_img = cv2.imread(cloud_path, cv2.IMREAD_ANYDEPTH) + cv2.imread(cloud_adj_path,cv2.IMREAD_ANYDEPTH)
        else:
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

            # IMPORTANT: convert landsat 8 band numbers to landsat 4,5,7 bands
            if(spacecraft_ID == 'LANDSAT_8'):
                band = convert_landsat8_band_to_landsat457_band(band)

            # skip if sr_band1 (430-450nm) on landsat 8 b/c it has no match for landsat 4,5,7
            if band == None:
                continue

            band_key = 'reflec_{}'.format(band)
            #print unwritten_bands
            #print band_key
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
    # TODO fix up description
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
    data = get_landsat_data(dict)
    print "Time to parse images and create dictionary: " + str(time.time() - before)

    before = time.time()

    # turn to pandas dataframe
    df = pd.DataFrame.from_dict(data)
    df = remove_data_where_cloud_cover_df(df)
    # write to csv
    df.to_csv('/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/landsat_data.csv', mode='wb+', index=False)
    print "Time to write dictionary to file: " + str(time.time() - before)


def time_and_write_landsat_usgs_spacial_data():
    # TODO fix up description
    '''
    Parse through every landsat image to extract the reflectances, cloud data, and meta data using the get_landsat_data function.
    Only get data where usgs mooring are located!
    Save data to csv.
    :return:
    '''
    before = time.time()
    dict = load_station_image_lat_long(
        path='/Users/Nathan/Desktop/Turbidity/SedimentLearning/data_formatter/usgs_moorings_locations.csv')

    data = get_landsat_data(dict)
    print 'Time to parse images and create dictionary corresponding to all landsat data at usgs locations: ' + str(
        time.time() - before)

    df = pd.DataFrame.from_dict(data)
    df.to_csv('/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/landsat_at_usgs_data.csv', mode='wb+', index=False)


def convert_all_usgs_to_UTC(paths=['/Users/Nathan/Dropbox/SedimentLearning/data/usgs/usgs_data_373015122071000.csv',
                                   '/Users/Nathan/Dropbox/SedimentLearning/data/usgs/usgs_data_374938122251801.csv'],
                            save_path='/Users/Nathan/Dropbox/SedimentLearning/data/usgs/usgs_data_373_374_UTC.csv'):
    '''
    Reads the polaris data csvs and returns one single pandas data frame containing the data. Add a datetime_UTC column.
    :param paths: paths to csvs
    :return: the dataframe. also saves to csv
    '''

    total_df = pd.DataFrame()
    for path in paths:
        df = pd.read_csv(path)
        name = path[path.rfind('_') + 1:path.rfind('.csv')]  # extract the name ie 373015122071000

        df['Station'] = name
        # replace tzinfo doesn't work with pst!
        # df['date_time_UTC'] = [datetime.strptime(x, '%Y-%m-%d %H:%M') \
        #                       .replace(tzinfo=pytz.timezone('US/Pacific')) for x in df['datetime']]

        pst = pytz.timezone('US/Pacific')

        df['date_time_UTC'] = [pst.localize(datetime.strptime(x, '%Y-%m-%d %H:%M')) for x in df['datetime']]
        df['date_time_UTC'] = [x.astimezone(pytz.utc) for x in df['date_time_UTC']]

        total_df = total_df.append(df)
        # print df.shape
    total_df.reset_index()
    # print total_df
    total_df.to_csv(path_or_buf=save_path)
    return total_df


# Filtered data produced on 3/3/16 may be incorrect because of timezone shifting
# TODO fix timezone error with converting pst to utc
# TODO polaris UTC data created on 3/3/16 off by about 1 hr!!! new code fixes this
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


    # set up pst timezone
    PST = pytz.timezone('US/Pacific')

    # strip spaces off of date_time
    # convert to date_time
    # localize to pst
    polaris['date_time_UTC'] = [PST.localize(datetime.strptime(x.strip(),'%m/%d/%Y %H%M')) for x in polaris.date_time_UTC]

    #convert to UTC
    polaris['date_time_UTC'] = [x.astimezone(pytz.timezone('UTC')) for x in polaris.date_time_UTC]

    # save to csv
    polaris.to_csv('/Users/Nathan/Dropbox/SedimentLearning/data/polaris/all_polaris_data_UTC.csv', mode='wb+',
                   index=False)
    print 'Time to execute conversion: ' + str(time.time() - before)
    return polaris


def convert_landsat_to_UTC(landsat_path = '/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/landsat_data.csv',
                           save_path = '/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/landsat_data_UTC.csv'):
    '''
    Sister function to convert_polaris_to_UTC. Reads the landsat data csv and converts gmt times to utc times.
    Then writes to a new csv and times execution.
    :return: return pandas dataframe
    '''
    before = time.time()
    data = pd.read_csv(landsat_path)
    data['date_time_UTC'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') \
                                 .replace(tzinfo=pytz.timezone('GMT')) for x in data['date_time']]
    data['date_time_UTC'] = [x.astimezone(pytz.timezone('UTC')) for x in data['date_time_UTC']]
    data.drop('date_time', axis=1, inplace=True)
    data.to_csv(save_path)
    print 'Time to execute conversion: ' + str(time.time() - before)
    return data


def read_usgs_to_df(usgs_utc_path='/Users/Nathan/Dropbox/SedimentLearning/data/usgs/usgs_data_373_374_UTC.csv'):
    return pd.read_csv(usgs_utc_path, low_memory=False)


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
        save_path_base='/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered',
        # IMPORTANT NO CSV EXTENSION
        filter_hours=24):
    '''
    Go through the landsat and polaris data and match each landsat data point to the closest (time wise) polaris data point.
    This is all done in a pandas dataframe which is then written to a csv and returned.
    :param landsat_utc_path:
    :param polaris_utc_path:
    :param save_path: path to save filtered csv to. IMPORTANT: THE CSV EXTENSION IS NOT
        INCLUDED SO I CAN TACK ON FILTER HOURS PARAM TO SAVE NAME

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

    # drop data where the reflectances are out of valid range: 0-10000
    for band in ['reflec_1', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_6', 'reflec_7']:
        filtered_df.drop(filtered_df.index[np.where(filtered_df[band] > 10000) or filtered_df[band] < 0], inplace=True)


    # remove data which is covered by clouds
    filtered_df = remove_data_where_cloud_cover_df(filtered_df)

    # write to csv
    save_path = save_path_base + '_' + str(filter_hours) + 'hr.csv'
    filtered_df.to_csv(save_path, mode='wb+', index=False)

    # return df
    return filtered_df


# TODO DOESN'T WORK: FIX USGS FILTERING NOT WORKING: WEIRD ERROR WITH DATAFRAME DATA TYPES NOT MATCHING AND EMPTY DATAFRAMES
# TODO may have something to do with how the usgs_data_373_374_UTC file doesn't include data for the 375 spot
def create_final_usgs_landsat_filtered_csv(
        landsat_utc_path='/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/landsat_at_usgs_data_UTC.csv',
        usgs_utc_path='/Users/Nathan/Dropbox/SedimentLearning/data/usgs/usgs_data_373_374_UTC.csv',
        save_path_base='/Users/Nathan/Dropbox/SedimentLearning/data/landsat_usgs_filtered/filtered',
        filter_hours=24):
    '''

    :param landsat_utc_path:
    :param usgs_utc_path:
    :param save_path_base:
    :param filter_hours:
    :return:
    '''
    # get dataframes
    landsat_df = read_landsat_to_df(landsat_utc_path)
    usgs_df = read_usgs_to_df(usgs_utc_path)

    # init filtered dataframe
    filtered_df = landsat_df.copy()
    filtered_df['time_diff'] = np.nan

    # drop unimportant usgs keys (keep only 05_80154 and datetime_UTC)
    usgs_df.drop(
        ['04_63680', '04_63680_cd', '04_80154', '04_80154_cd', '04_63680', '04_63680_cd', '04_80154', '04_80154_cd',
         '05_80154_cd', '06_63680', '06_63680_cd', '07_63680', '07_63680_cd', 'datetime', 'tz_cd'],axis=1,inplace=True)

    # put all the polaris keys into the filtered df
    for key in usgs_df.keys():
        if key not in landsat_df.keys():
            filtered_df[key] = np.nan

    # convert the date_time string representations to actual datetime objects
    # Important: because %z (utc offset string) is not a supported directive on python 2.7 I truncate the utc offset which is 0
    landsat_df['date_time_UTC'] = [datetime.strptime(x[:-6], '%Y-%m-%d %H:%M:%S') for x in landsat_df['date_time_UTC']]
    usgs_df['date_time_UTC'] = [datetime.strptime(x[:-6], '%Y-%m-%d %H:%M:%S') for x in usgs_df['date_time_UTC']]

    # Set the landsat_utc column
    filtered_df['landsat_UTC'] = landsat_df['date_time_UTC']
    filtered_df['usgs_UTC'] = np.nan

    landsat_df['station_ID'] = landsat_df['station_ID'].astype(str)
    usgs_df['Station'] = usgs_df['Station'].astype(str)


    # loop through each polaris station (data collection site)
    # TODO last usgs not included in usgs df
    for station in landsat_df['station_ID'].unique():
        if station not in usgs_df['Station']:
            continue

        # create a new dataframe with only data from that location
        landsat_subset_df = landsat_df[landsat_df['station_ID'] == station]
        #print landsat_subset_df['station_ID']
        #print usgs_df['Station']

        # TODO fix inconsistency between usgs's 'station' identifier and polaris' 'station number' identifier
        # TODO fix this expression it isn't working, mismatched data types?
        # TODO fix how digits were getting cut off!
        usgs_subset_df = usgs_df[usgs_df['Station'] == station]
        print 'usgs subset'
        print usgs_subset_df

        print 'landsat subset'
        print landsat_subset_df

        # counts how many data points we've written for this station
        count = 0

        # loop through each data point in the landsat subset
        for idx, date_time in zip(landsat_subset_df.index, landsat_subset_df['date_time_UTC']):
            # print progress
            print '{} / {} for location {}'.format(count, len(landsat_subset_df), station)
            count += 1

            # calculate and copy over time difference
            #print usgs_subset_df
            time_diff = np.abs(date_time - usgs_subset_df['date_time_UTC'])

            # find index of smallest time difference
            index_smallest = np.argmin(time_diff)
            filtered_df.loc[idx, 'time_diff'] = time_diff[index_smallest]

            # copy polaris data over to landsat data
            for key in usgs_df.keys():
                if key == 'date_time_UTC': break
                filtered_df.loc[idx, key] = usgs_subset_df.loc[index_smallest, key]

            # copy polaris date_time
            filtered_df.loc[idx, 'usgs_UTC'] = usgs_subset_df.loc[index_smallest, key]

    # drop the date_time_UTC column b/c it is equiv to landsat_UTC column
    filtered_df.drop('date_time_UTC', axis=1, inplace=True)

    print filtered_df

    # drop the date time columns from polaris data
    # filtered_df.drop('Date', axis=1, inplace=True)
    # filtered_df.drop('Time', axis=1, inplace=True)

    # IMPORTANT
    # filter data for time differences < filter_hours parameter
    filtered_df = filtered_df[filtered_df.time_diff < timedelta(hours=filter_hours)]

    # drop data where the reflectances are out of valid range: 0-10000
    for band in ['reflec_1', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_6', 'reflec_7']:
        filtered_df.drop(filtered_df.index[np.where(filtered_df[band] > 10000)], inplace=True)

    # write to csv
    save_path = save_path_base + '_' + str(filter_hours) + 'hr.csv'
    filtered_df.to_csv(save_path, mode='wb+', index=False)

    # return df
    return filtered_df

def create_varied_cutoff_csvs(
        save_path_base='/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered',
        cutoffs=[1, 2, 4, 8, 12, 16, 20, 24]):
    '''
    Uses the filtered landsat/polaris csv with the 24 hr cutoff to create new csvs from it with different time cutoffs
    :param save_path_base: base file path without the cutoff or csv extension
    :param cutoffs: list of cutoffs
    :return: nothing
    '''
    df = pd.read_csv(save_path_base + '_24hr.csv')

    df['landsat_UTC'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df['landsat_UTC']]
    df['polaris_UTC'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df['polaris_UTC']]
    df['time_diff'] = np.abs(df['landsat_UTC'] - df['polaris_UTC'])

    # drop data where the reflectances are out of valid range: 1-10000
    for band in ['reflec_1', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_6', 'reflec_7']:
        df.drop(df.index[np.where(df[band] > 10000)], inplace=True)

    for cutoff in cutoffs:
        filtered_df = df[df.time_diff < timedelta(hours=cutoff)]
        filtered_df.to_csv(save_path_base + '_' + str(cutoff) + 'hr.csv', index=False)

if __name__ == '__main__':
    data_columns = ['date_time', 'cf_mask_quality', 'cloud', 'station_ID', 'lat', 'long', 'landsat_scene', 'reflec_1',
                    'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_6', 'reflec_7']


    #### workflow for creating landsat/polaris data
    # get_image_xy_in_scene_for_lat_long(path = '/Users/Nathan/Desktop/Turbidity/SedimentLearning/data_formatter/station_img_coordinates.csv')
    # time_and_write_landsat_data()
    # convert_polaris_to_UTC()
    # convert_landsat_to_UTC()
    filtered = create_final_filtered_csv()
    print filtered
    create_varied_cutoff_csvs()


    #### workflow for matching landsat data with usgs data:
    # get_usgs_moorings_image_lat_long()
    # convert_all_usgs_to_UTC()
    # time_and_write_landsat_usgs_spacial_data()
    # convert_landsat_to_UTC(landsat_path='/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/landsat_at_usgs_data.csv',
    #                        save_path='/Users/Nathan/Desktop/Turbidity/SedimentLearning/data/landsat_at_usgs_data_UTC.csv')
    # create_final_usgs_landsat_filtered_csv()
