import regression
import convex as mycvx
import numpy as np
import data_formatter.landsat_get_data as lgd
import cv2
import glob
import pandas as pd


def print_theta():
    # TODO write this
    return 0


def get_feature_array(scene_folder_path):
    # TODO: fill this out
    '''
    Load the images in the scene folder into an array that the model can read. The format is:

    :param scene_folder_path:
    :return:
    '''

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

    assert np.array_equal(full_data[:, 0], data['reflec_1'])
    print('Loaded surface reflectance tifs. Calculating band ratios...')

    top_5_band_ratios_array = regression.top_5_band_ratios(full_data)
    print 'Sum square differences: ' + str(np.sum(np.power(full_data[:, 0] - data['reflec_1'], 2)))
    assert np.array_equal(full_data[:, 0], data['reflec_1'])
    # Add to feature array
    full_data = np.append(full_data, top_5_band_ratios_array, axis=1)

    print('Finished loading regression features: 5 band ratios + 6 surface reflectances')

    print 'Sum square differences: ' + str(np.sum(np.power(full_data[:, 0] - data['reflec_1'], 3)))
    # print full_data[:,0] - data['reflec_1']
    assert np.array_equal(full_data[:, 0], data['reflec_1'])

    return full_data, image_shape


def create_model():
    print('Training robust regression')
    x, y = regression.get_data(
        filenames=['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_2hr.csv'],
        spm_cutoff=None)  # 2hr data

    # Get top 5 correlated band ratios
    top_5_bands = regression.top_5_band_ratios(x)
    # Add to feature array
    x = np.append(x, top_5_bands, axis=1)

    '''
    # Get top 3 correlated band ratios
    top_3_bands = regression.division_feature_expansion(x)[:,[29,9,14,28,5]]
    # Add to feature array
    x = np.append(x,top_3_bands,axis=1)
    '''

    # log spm regression
    logy = np.log(y)
    alpha = 8
    seed = 4
    model = mycvx.kfolds_convex(x, logy, alpha, random_seed=seed)
    theta = model['theta']
    print('Done training robust regression. Starting SPM prediction calculation...')

    return theta


def create_color_map(scene_path=''):
    '''

    :param scene_path:
    :return:
    '''
    scene_data, image_shape = get_feature_array(scene_path)

    color_img = np.zeros((image_shape[0], image_shape[1], 3))
    color_img[:, :, 0] = scene_data[:, 0].reshape(image_shape)  # blue - band 1
    color_img[:, :, 1] = scene_data[:, 1].reshape(image_shape)  # green - band 2
    color_img[:, :, 2] = scene_data[:, 2].reshape(image_shape)  # red - band 3

    # scale values
    color_img = color_img * 255. / 4000.

    all_imgs = lgd.get_scene_imgs(scene_path)
    scene_name = lgd.get_scene_name(all_imgs[0])
    cv2.imwrite('../figures/color_map_{}.jpg'.format(scene_name), color_img)


def create_spm_map(theta=None, scene_path='', log_spm_flag=True, color_flag=True):
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

    scene_data, image_shape = get_feature_array(scene_path)

    predicted_spm_log = np.dot(scene_data, theta)
    predicted_spm = np.exp(predicted_spm_log)

    print('Done creating predicted SPM map')

    if (log_spm_flag):
        spm_map = predicted_spm_log.reshape(image_shape)
        spm_map[spm_map > 4] = 4
        spm_map[spm_map < 0] = 0

        # high log(spm) values are around 4 so 256/4 =64
        spm_map = spm_map * 64
    else:
        spm_map = predicted_spm.reshape(image_shape)
        # spm_map[spm_map>500] = 500
        spm_map[spm_map < 0] = 0

        # high spm values are around 150
        # map 0-150 to 0-256
        # spm_map = spm_map * 256./150.

    # make darker areas correspond to higher turbidity
    spm_map = 255 - np.array(spm_map, dtype='uint8')

    all_imgs = lgd.get_scene_imgs(scene_path)

    if (color_flag):
        # want water to be grayscale
        # want land to be green
        # want clouds to be blue
        # clouds on top of land (blue instead of green)

        spm_map_color = np.zeros((image_shape[0], image_shape[1], 3))
        spm_map_color[:, :, 0] = spm_map  # red
        spm_map_color[:, :, 1] = spm_map  # green - land
        spm_map_color[:, :, 2] = spm_map  # bue - cloud

        cloud_path = lgd.get_cloud(all_imgs)
        print cloud_path
        cloud = cv2.imread(cloud_path, cv2.IMREAD_ANYDEPTH)

        cloud = np.array((cloud / 255.), dtype='uint8')

        water = cv2.imread(
            '/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LT50440342010098-SC20160218112217/LT50440342010098PAC01_sr_land_water_qa.tif',
            cv2.IMREAD_ANYCOLOR)
        water = np.array((water / 255.), dtype='uint8')

        # 1 if water, 0 if not water
        spm_map_color[:, :, 0][water == 0] = 0
        spm_map_color[:, :, 2][water == 0] = 0
        spm_map_color[:, :, 1][water == 0] += 100  # green
        # let green channel be equal to spm to shade land mass

        spm_map_color[:, :, 0][cloud == 1] = 0
        spm_map_color[:, :, 1][cloud == 1] = 0
        spm_map_color[:, :, 2][cloud == 1] += 100  # red
        # let blue channel be equal to spm to shade cloud mass

    df = pd.DataFrame(spm_map.ravel())
    print df.describe()

    scene_name = lgd.get_scene_name(all_imgs[0])
    log_str = ('log_' if log_spm_flag else '')
    color_str = ('_color' if color_flag else '')
    cv2.imwrite('../figures/{}spm_map{}_{}.jpg'.format(log_str, color_str, scene_name),
                spm_map_color if color_flag else spm_map)


if __name__ == '__main__':
    # get_image_matrix(scene_folder_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LE70440342003007-SC20160218112750/')
    theta = create_model()
    # create_spm_map(theta,scene_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LE70440342003007-SC20160218112750/')
    # create_spm_map(theta,scene_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LT50440342009271-SC20160218112659/')
    # create_spm_map(theta,scene_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LT50440342010194-SC20160218111641/')
    # create_spm_map(theta,scene_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LE70440342003071-SC20160218112128/')
    # create_spm_map(theta,scene_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LE70440342003007-SC20160218112750/')

    two_hr_scenes = ['LE70440342003007EDC00', 'LE70440342003055EDC00', 'LE70440342003071EDC00', 'LE70440342012080EDC00',
                     'LE70440342012144EDC00', 'LE70440342012240EDC00', 'LE70440342014133EDC00', 'LE70440342015072EDC00',
                     'LT50440342007234PAC01', 'LT50440342008253PAC01', 'LT50440342009079PAC01', 'LT50440342009239PAC01',
                     'LT50440342009271PAC01', 'LT50440342010194PAC01', 'LT50440342010274PAC01', 'LT50440342011165PAC02']
    # two_hr_scenes = ['LE70440342003055EDC00']
    # two_hr_scenes = ['LT50440342007234PAC01']

    for scene in two_hr_scenes:
        path = glob.glob('/Users/Nathan/Dropbox/SedimentLearning/data/landsat/' + scene[:-5] + '*')[0] + '/'
        print path
        # create_spm_map(theta, scene_path=path, log_spm_flag=True)
        create_color_map(scene_path=path)
