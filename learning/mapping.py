import regression
import convex as mycvx
import numpy as np
import data_formatter.landsat_get_data as lgd
import cv2
import glob
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize


def get_feature_array_L57(scene_folder_path):
    # TODO: fill this out
    '''
    ONLY WORKS FOR LANDSAT 5 and 7 images. Landsat 8 images have different band wavelengths

    Load the images in the scene folder into an array that the model can read. The format is:
    ['reflec_1', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_7', ratio1, ratio2, ratio3, ratio4, ratio5]

    :param scene_folder_path: full path to folder of scene
    :return:
    '''
    print 'Getting features from scene images'
    # data to load
    data_columns = ['reflec_1', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_7']
    data = {}
    for key in data_columns:
        data[key] = np.array([])

    image_shape = np.nan

    # get names of all images in the scene folder
    imgs = lgd.get_scene_imgs(scene_folder_path)
    # get the paths for the different types of image
    sr_paths = lgd.get_sr_band_imgs(imgs)

    for path, band in zip(sr_paths, data_columns):
        # print path, band
        data[band] = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

        if image_shape is np.nan:
            image_shape = data[band].shape
            # print image_shape

        data[band] = data[band].ravel()

    full_data = np.zeros((6, data['reflec_1'].size))

    # put data in correct order
    full_data[0] = data['reflec_1']
    full_data[1] = data['reflec_2']
    full_data[2] = data['reflec_3']
    full_data[3] = data['reflec_4']
    full_data[4] = data['reflec_5']
    full_data[5] = data['reflec_7']

    # Transpose the data
    full_data = full_data.T

    # make sure the first column is reflec 1
    assert np.array_equal(full_data[:, 0], data['reflec_1'])

    print('Loaded surface reflectance tifs. Calculating band ratios...')

    full_data = regression.Kau_MB_BR_features(full_data)

    # make sure the first column is still reflec 1
    # print 'Sum square differences: ' + str(np.sum(np.power(full_data[:, 0] - data['reflec_1'], 2)))
    assert np.array_equal(full_data[:, 0], data['reflec_1'])

    print('Finished loading regression features: 3 band ratios + 4 surface reflectances')

    # create matrix representing valid data
    valid_data = np.ones_like(data['reflec_1'])
    for band in range(4):
        valid_data[full_data[:, band] < 0] = 0
        valid_data[full_data[:, band] > 10000] = 0

    return full_data, image_shape, valid_data


def create_model():
    print('Training robust regression')
    x, y = regression.get_data(
        filenames=['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_2hr.csv'],
        spm_cutoff=None)  # 2hr data

    # Get top 5 correlated band ratios and add to feature array
    x = regression.Kau_MB_BR_features(x)
    # Shape of x is (75,11)

    # log spm regression
    logy = np.log(y)
    alpha = 8
    seed = 4
    model = mycvx.kfolds_convex(x, logy, alpha, random_seed=seed)
    theta = model['theta']

    y_test = model['data']['y_test']
    y_pred = model['data']['y_pred']
    y_train = model['data']['y_train']
    y_train_pred = model['data']['y_train_pred']

    r2_test = np.round(r2_score(np.exp(y_test), np.exp(y_pred)), 3)
    r2_train = np.round(r2_score(np.exp(y_train), np.exp(y_train_pred)), 3)

    print(
    'Done training robust regression. R2 of actual spm vs predicted spm on training set = {}. \n'.format(r2_train))

    return theta


def create_color_map(scene_path=''):
    '''
    Uses the rgb bands to reconstruct a 'true color' satellite image
    :param scene_path:
    :return:
    '''
    scene_data, image_shape, valid = get_feature_array_L57(scene_path)

    color_img = np.zeros((image_shape[0], image_shape[1], 3))
    color_img[:, :, 0] = scene_data[:, 0].reshape(image_shape)  # blue - band 1
    color_img[:, :, 1] = scene_data[:, 1].reshape(image_shape)  # green - band 2
    color_img[:, :, 2] = scene_data[:, 2].reshape(image_shape)  # red - band 3

    # scale values
    color_img = color_img * 255. / 4000.

    all_imgs = lgd.get_scene_imgs(scene_path)
    scene_name = lgd.get_scene_name(all_imgs[0])
    cv2.imwrite('../figures/color_map_{}.jpg'.format(scene_name), color_img)


def create_spm_map(theta=None, scene_path='', log_spm_flag=False, color_flag=True):
    # TODO add actual spm values to legend
    '''
    For info on sr_cloud_qa and sr_land_water_qa image values
    http://landsat.usgs.gov/landsat_climate_data_records_quality_calibration.php
    cloud: 255 = cloud, water: 255 = water

    :param theta: theta for the linear model
    :param scene_path: full path to the scene folder
    :param log_spm_flag: flag for whether to map spm or log(spm), true means log(spm)
    :return:
    '''
    if (theta is None):
        theta = create_model()
    else:
        print('Using given theta. Creating predicted SPM map')

    scene_data, image_shape, valid = get_feature_array_L57(scene_path)

    predicted_spm_log = np.dot(scene_data, theta)
    predicted_spm = np.exp(predicted_spm_log)

    # map spm map to -1 to 1 range
    if (log_spm_flag):
        spm_map = predicted_spm_log.reshape(image_shape)

        # high log(spm) values are around 4 so map (-1)-4 to -1-1
        spm_map = np.interp(spm_map, (-1, 4), (-1, 1))
    else:
        spm_map = predicted_spm.reshape(image_shape)

        # high spm values are around 50
        # map 0-50 to -1 to 1
        spm_map = np.interp(spm_map, (0, 50), (-1, 1))

    print('Done creating predicted SPM map')

    # map spm to jet color scale, exclude the alpha channel
    spm_map = plt.cm.jet(spm_map)[:, :, 0:3]

    # color code the map: make land black, make cloud white
    spm_map = color_map_by_land_water(spm_map, scene_path)

    print 'Done color coding map'

    # create figure
    dpi = 400
    plt.figure(dpi=dpi)
    plt.imshow(spm_map, interpolation='nearest')
    c = plt.colorbar(ticks=[0, 1.0])

    # add labels depending on log or not
    if (log_spm_flag):
        c.set_ticklabels(['ln(SPM) < -1', 'ln(SPM) > 4'])
    else:
        c.set_ticklabels(['SPM < 0', 'SPM > 50'])

    # figure out the name
    all_imgs = lgd.get_scene_imgs(scene_path)
    scene_name = lgd.get_scene_name(all_imgs[0])
    log_str = ('log_' if log_spm_flag else '')
    folder = log_str + 'colormap_spm'

    datetime = lgd.get_datetime_from_metadata(lgd.get_metadata_path(scene_path))

    # Add title
    plt.title('Map of SPM in SF Bay: ' + str(scene_name) + '\n' + str(datetime) + ' GMT')

    # save figure
    plt.savefig('../figures/{}/{}{}.jpg'.format(folder, log_str, scene_name), dpi=dpi)
    print 'Done saving map\n'


def color_map_by_land_water(spm_map, scene_path):
    # color code image by cloud and land
    all_imgs = lgd.get_scene_imgs(scene_path)

    # read cloud image
    cloud_path = lgd.get_cloud(all_imgs)
    cloud = cv2.imread(cloud_path, cv2.IMREAD_ANYDEPTH)
    cloud = np.array((cloud / 255.), dtype='uint8')

    # make clouds white
    spm_map[cloud == 1] = 1.0

    # read water image
    water = cv2.imread(
        '/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LT50440342010098-SC20160218112217/LT50440342010098PAC01_sr_land_water_qa.tif',
        cv2.IMREAD_ANYCOLOR)
    water = np.array((water / 255.), dtype='uint8')

    # make land mass black
    spm_map[water == 0] = 0

    # make land and cloud gray
    spm_map[np.multiply((1 - water), cloud) == 1] = 0.8

    return spm_map


if __name__ == '__main__':
    theta = create_model()
    # create_spm_map(theta,scene_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LE70440342003007-SC20160218112750/')

    two_hr_scenes = ['LE70440342003007EDC00', 'LE70440342003055EDC00', 'LE70440342003071EDC00', 'LE70440342012080EDC00',
                     'LE70440342012144EDC00', 'LE70440342012240EDC00', 'LE70440342014133EDC00', 'LE70440342015072EDC00',
                     'LT50440342007234PAC01', 'LT50440342008253PAC01', 'LT50440342009079PAC01', 'LT50440342009239PAC01',
                     'LT50440342009271PAC01', 'LT50440342010194PAC01', 'LT50440342010274PAC01', 'LT50440342011165PAC02']
    # two_hr_scenes = ['LE70440342003055EDC00']
    # two_hr_scenes = ['LT50440342007234PAC01']

    # two options for which maps to make
    # only scenes used for training:
    folder_paths = [glob.glob('/Users/Nathan/Dropbox/SedimentLearning/data/landsat/' + scene[:-5] + '*')[0] + '/' for scene in two_hr_scenes]

    # all landsat 5 scenes (landsat 7 scenes have diag bands)
    # folder_paths = [path + '/' for path in glob.glob('/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LT5*')]
    print folder_paths

    for path in folder_paths:
        print 'Current scene\'s folder path: ' + str(path)
        create_spm_map(theta, scene_path=path, log_spm_flag=True)
        # create_color_map(scene_path=path)
