import regression
import convex as mycvx
import numpy as np
from data_formatter.landsat_get_data import get_scene_imgs,get_sr_band_imgs,get_scene_name
import cv2


def get_feature_array(scene_folder_path):
    data_columns = [ 'reflec_1','reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_7']
    data = {}
    for key in data_columns:
        data[key] = np.array([])

    image_shape = np.nan

    # get names of all images in the scene folder
    imgs = get_scene_imgs(scene_folder_path)
    # get the paths for the different types of image
    sr_paths = get_sr_band_imgs(imgs)
    for path,band in zip(sr_paths,data.keys()):
        data[band] = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

        if image_shape is np.nan:
            image_shape = data[band].shape
            # print image_shape

        data[band] = data[band].ravel()

    full_data = np.zeros((6,data['reflec_1'].size))

    full_data[0] = data['reflec_1']
    full_data[1] = data['reflec_2']
    full_data[2] = data['reflec_3']
    full_data[3] = data['reflec_4']
    full_data[4] = data['reflec_5']
    full_data[5] = data['reflec_7']

    full_data = full_data.T
    print('Loaded surface reflectance tifs. Calculating band ratios...')

    top_5_band_ratios = regression.top_5_band_ratios(full_data)
    # Add to feature array
    full_data = np.append(full_data,top_5_band_ratios,axis=1)

    print('Finished loading regression features: 5 band ratios + 6 surface reflectances')

    return full_data,image_shape
def create_model():
    print('Training robust regression')
    x,y=regression.get_data(filenames=['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_2hr.csv'],spm_cutoff=-1) # 2hr data

    # Get top 5 correlated band ratios
    top_5_bands = regression.top_5_band_ratios(x)
    # Add to feature array
    x = np.append(x,top_5_bands,axis=1)

    '''
    # Get top 3 correlated band ratios
    top_3_bands = regression.division_feature_expansion(x)[:,[29,9,14,28,5]]
    # Add to feature array
    x = np.append(x,top_3_bands,axis=1)
    '''

    # log spm regression
    logy = np.log(y)
    alpha = 8
    model = mycvx.kfolds_convex(x, logy, alpha)
    theta = model['theta']
    print('Done training robust regression. Starting SPM prediction calculation...')

    return theta
def create_spm_map(theta = None,scene_path = ''):
    if(theta is None):
        theta = create_model()
    else:
        print('Using given theta. Creating predicted SPM map')

    scene_data, image_shape = get_feature_array(scene_path)
    # print scene_data

    predicted_spm = np.exp(np.dot(scene_data,theta))
    print('Done creating predicted SPM map')

    # print predicted_spm,predicted_spm.shape
    # print image_shape

    spm_map = predicted_spm.reshape(image_shape)
    spm_map[spm_map>500] = 500
    spm_map[spm_map<0] = 0

    # spm_map = spm_map * 256./200.

    # print spm_map
    spm_map = np.array(spm_map,dtype='uint8')

    # spm_map_color = np.zeros()

    scene_name = get_scene_name(get_scene_imgs(scene_path)[0])
    cv2.imwrite('../figures/spm_map_{}.png'.format(scene_name),spm_map)

if __name__ == '__main__':
    # get_image_matrix(scene_folder_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LE70440342003007-SC20160218112750/')
    theta = create_model()
    create_spm_map(theta,scene_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LE70440342003007-SC20160218112750/')
    create_spm_map(theta,scene_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LT50440342009271-SC20160218112659/')
    create_spm_map(theta,scene_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LT50440342010194-SC20160218111641/')
    create_spm_map(theta,scene_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LE70440342003071-SC20160218112128/')
    create_spm_map(theta,scene_path='/Users/Nathan/Dropbox/SedimentLearning/data/landsat/LE70440342003007-SC20160218112750/')

