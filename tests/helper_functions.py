import csv
import numpy as np
import test_lin_reg
from os import listdir
from os.path import isfile, join
import cv2
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

def get_AOI(img_path,(img_lat0,img_long0),(img_lat1,img_long1),(aoi_lat0,aoi_long0),(aoi_lat1,aoi_long1)=(None,None),radius = None):
    # 0 is the top left, 1 is the bottom right
    # for some reason it's usually read lat,long not long, lat
    #img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

    img = cv2.imread(img_path,cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise IOError
    print 'Processed image'

    img_height,img_width = img.shape # like a matrix so height first

    # return a square of pixels around the area of interest (aoi) with a square size = 2*radius
    if aoi_lat1 is None:
        # interpolate the image coordinates
        [img_aoi_x] = np.int32(np.interp([aoi_long0],[img_long0,img_long1],[0,img_width]))
        [img_aoi_y] = np.int32(np.interp([aoi_lat0],[img_lat1,img_lat0],[img_height,0]))
        if img_lat1 > img_lat0:
            raise ValueError # interp doesn't work in this case

        print 'AOI point: {}'.format((img_aoi_x,img_aoi_y))

        # a square of pixels
        if radius > 0:
             # annoying min max so image isn't out of bounds
            return img[max(img_aoi_y-radius,0):min(img_aoi_y+radius,img_height),
                       max(img_aoi_x-radius,0):min(img_aoi_x+radius,img_width)]
        # just one pixel
        elif radius == 0:
            # put the sr value into an array so i can display it with imshow
            return np.array([[img[img_aoi_y,img_aoi_x]]])
        raise ValueError # radius shouldn't be None (you have to set it)

    # return the specified area in the image
    else:
        # interpolate the image coordinates
        img_aoi_left,img_aoi_right = np.int32(np.interp([aoi_long0,aoi_long1],[img_long0,img_long1],[0,img_width]))
        img_aoi_top,img_aoi_bottom = np.int32(np.interp([aoi_lat0,aoi_lat1],[img_lat1,img_lat0],[img_height,0]))
        #print 'aoi image left right: {},{}'.format(img_aoi_left,img_aoi_right)
        return img[img_aoi_top:img_aoi_bottom,img_aoi_left:img_aoi_right]

if __name__ == '__main__':
    data_path = '/Users/Nathan/Dropbox/SedimentLearning/data/full/'
    get_station_locations(data_path)

    meta_path = '/Users/Nathan/Desktop/Turbidity/SedimentLearning/tests/all_scenes/'
    #get_scenenames_from_metadata(meta_path)

    ''' META DATA
    <lpgs_metadata_file>LC80440342013298LGN00_MTL.txt</lpgs_metadata_file>
        <corner location="UL" latitude="38.528160" longitude="-123.415310"/>
        <corner location="LR" latitude="36.397470" longitude="-120.841200"/>

          <bounding_coordinates>
            <west>-123.415482</west>
            <east>-120.779472</east>
            <north>38.529033</north>
            <south>36.397332</south>
        </bounding_coordinates>
        '''
    ''' bay location: 37.462640, -122.049717
    tip of sf 37.811068, -122.477161'''
    #img = get_AOI('/Users/Nathan/Desktop/Turbidity/SedimentLearning/tests/LC80440342013298LGN00_band11.tif',(38.528160,-123.415310),(36.397470,-120.841200),(37.462640, -122.049717),radius=300)
    img = get_AOI('/Users/Nathan/Desktop/Turbidity/SedimentLearning/tests/LC80440342013298LGN00_sr_evi.tif',(38.529033,-123.415482),(36.397332,-120.779472),(37.811068, -122.477161),radius=100)

    #draw crosshairs
    h,w = img.shape
    cv2.line(img,(w/2,0),(w/2,h),(255,255,255))
    cv2.line(img,(0,h/2),(w,h/2),(255,255,255))

    print img
    print img.shape
    cv2.imshow('hello',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ''' major problem
        image is off, not finding the right AOI, by about 10 pixels = .1% error in projection?
    '''