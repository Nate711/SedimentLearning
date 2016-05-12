import regression
import convex as mycvx
import numpy as np
import data_formatter.landsat_get_data as lgd
import cv2
import glob
import pandas as pd
from sklearn.metrics import r2_score

def get_feature_array(scene_folder_path):
    # TODO: fill this out
    '''
    Load the images in the scene folder into an array that the model can read. The format is:
    ['reflec_1', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_7', ratio1, ratio2, ratio3, ratio4, ratio5]

    :param scene_folder_path: full path to folder of scene
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

    # make sure the first column is reflec 1
    assert np.array_equal(full_data[:, 0], data['reflec_1'])
    print('Loaded surface reflectance tifs. Calculating band ratios...')

    # Add band ratios to feature array
    top_5_band_ratios_array = regression.top_5_band_ratios(full_data)
    full_data = np.append(full_data, top_5_band_ratios_array, axis=1)

    # make sure the first column is still reflec 1
    # print 'Sum square differences: ' + str(np.sum(np.power(full_data[:, 0] - data['reflec_1'], 2)))
    assert np.array_equal(full_data[:, 0], data['reflec_1'])


    print('Finished loading regression features: 5 band ratios + 6 surface reflectances')

    # Remove plain reflectances for testing of the white splotches
    # full_data = full_data[:,6:]

    # Remove specific columns in full_data to test where white splotches are coming from
    # full_data[:,10] = 0

    # Test on only one band
    # full_data = full_data[:,1:2]

    # create matrix representing valid data
    valid_data = np.ones_like(data['reflec_1'])
    for band in range(6):
        valid_data[full_data[:,band] < 0] = 0
        valid_data[full_data[:,band] > 10000] = 0

    return full_data, image_shape, valid_data


def create_model():
    print('Training robust regression')
    x, y = regression.get_data(
        filenames=['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_2hr.csv'],
        spm_cutoff=None)  # 2hr data

    # Get top 5 correlated band ratios
    top_5_bands = regression.top_5_band_ratios(x)

    # Add to feature array
    x = np.append(x, top_5_bands, axis=1)
    # Shape of x is (75,11)

    # Remove specific columns in x to test where white splotches are coming from
    # x[:,10] = 0

    # Remove plain reflectances for testing of the white splotches
    # x = x[:,6:]

    # Test on only one band
    # x = x[:,1:2]

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

    print('Done training robust regression. True scale R2 train = {}. Starting SPM prediction calculation...'.format(r2_train))

    return theta


def create_color_map(scene_path=''):
    '''
    Uses the rgb bands to reconstruct a 'true color' satellite image
    :param scene_path:
    :return:
    '''
    scene_data, image_shape, valid = get_feature_array(scene_path)

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

    scene_data, image_shape, valid = get_feature_array(scene_path)

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
        # print 'not log'
        spm_map = predicted_spm.reshape(image_shape)
        # spm_map[spm_map>500] = 500
        spm_map[spm_map < 0] = 0

        # high spm values are around 150
        # map 0-50 to 0-256
        spm_map = spm_map * 256./50.

    # make darker areas correspond to higher turbidity
    spm_map = 255 - np.array(spm_map, dtype='uint8')


    if (color_flag):
        # want water to be grayscale
        # want land to be green
        # want clouds to be blue
        # clouds on top of land (blue instead of green)

        spm_map_color = np.zeros((image_shape[0], image_shape[1], 3))
        spm_map_color[:, :, 0] = spm_map  # red
        spm_map_color[:, :, 1] = spm_map  # green - land
        spm_map_color[:, :, 2] = spm_map  # bue - cloud

        # color code image by cloud and land
        all_imgs = lgd.get_scene_imgs(scene_path)
        cloud_path = lgd.get_cloud(all_imgs)
        print 'Cloud tif path: ' + str(cloud_path)
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
        # let red channel be equal to spm to shade cloud mass

        # color code non-valid data spots as red
        valid_shaped = valid.reshape(image_shape)
        spm_map_color[:, :, 0][valid_shaped == 0] = 0
        spm_map_color[:, :, 1][valid_shaped == 0] = 0
        spm_map_color[:, :, 2][valid_shaped == 0] += 100  # red

    # Convert image to dataframe and print numerical summary of spm data (exluding land spm)
    df = pd.DataFrame(spm_map[water == 1].ravel())
    print df.describe()

    # Draw SPM legend!
    cv2.rectangle(spm_map_color,(2200,2200),(3066,2511),(0,0,0),thickness=-1)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(spm_map_color,'Black = >50 mg/L',(2250,2300),font,2,(255,255,255),thickness=4)
    cv2.putText(spm_map_color,'White = 0 mg/L',(2250,2400),font,2,(255,255,255),thickness=4)


    scene_name = lgd.get_scene_name(all_imgs[0])
    log_str = ('log_' if log_spm_flag else '')
    color_str = ('_color' if color_flag else '')
    folder = log_str + 'spm_maps' + ('_color' if color_flag else '')
    cv2.imwrite('../figures/{}/{}spm_map{}_{}.jpg'.format(folder,log_str, color_str, scene_name),
                spm_map_color if color_flag else spm_map)


if __name__ == '__main__':
    theta = create_model()
    # create_spm_map(theta,scene_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LE70440342003007-SC20160218112750/')


    two_hr_scenes = ['LE70440342003007EDC00', 'LE70440342003055EDC00', 'LE70440342003071EDC00', 'LE70440342012080EDC00',
                     'LE70440342012144EDC00', 'LE70440342012240EDC00', 'LE70440342014133EDC00', 'LE70440342015072EDC00',
                     'LT50440342007234PAC01', 'LT50440342008253PAC01', 'LT50440342009079PAC01', 'LT50440342009239PAC01',
                     'LT50440342009271PAC01', 'LT50440342010194PAC01', 'LT50440342010274PAC01', 'LT50440342011165PAC02']
    # two_hr_scenes = ['LE70440342003055EDC00']
    two_hr_scenes = ['LT50440342007234PAC01']

    for scene in two_hr_scenes:
        path = glob.glob('/Users/Nathan/Dropbox/SedimentLearning/data/landsat/' + scene[:-5] + '*')[0] + '/'
        print path
        create_spm_map(theta, scene_path=path, log_spm_flag=True)
        # create_color_map(scene_path=path)
