import time

__author__ = 'jadelson'

import urllib2
import netCDF4 as nc
import multiprocessing as mp
import csv
import numpy as np
import os

locations = {}
cc_names = []

def download_usgs_data(site_id, begin_date, end_date, data_dir='/Users/jadelson/Desktop/'):
    """
    Extracts timeseries data from the USGS nwis water data service.

    :param site_id:  USGS station ID
    :param begin_date: Query beginning date as string format 'yyyy-mm-dd'
    :param end_date: Query ending date as string format 'yyyy-mm-dd'
    :return:
    """
    target_url = "http://nwis.waterdata.usgs.gov/ca/nwis/uv?cb_80154=on&cb_63680=on&format=rdb&site_no=%s&period" \
                 "=&begin_date=%s&end_date=%s" % (site_id, begin_date, end_date)

    with open(data_dir + 'usgs_data_%s.csv' % site_id, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        print target_url
        # print csvfile
        try:
            print 'Opening: %s' % target_url
            f = urllib2.urlopen(target_url)
        except StandardError:
            print 'cannot open url: \n %s' % target_url
            return
        for s in f:
            s = s.split('\n')
            line = s[0].split('\t')
            if line[0][0] != '#' and line[0] != '5s':
                csvwriter.writerow(line[2:])


def get_usgs_list(usgs_location_filename):
    """
    Loads the list of USGS stations from the usgs location .csv file

    :param usgs_location_filename:
    :return:
    """
    csv_file = open(usgs_location_filename, 'r')
    locations = {}
    for row in csv_file:
        r = row.split(',')
        station_id = r[0]
        latitude = float(r[1]) + float(r[2]) / 60. + float(r[3]) / 3600.
        longitude = -(float(r[4]) + float(r[5]) / 60. + float(r[6]) / 3600.)
        locations[station_id] = {'long': longitude, 'lat': latitude}
    return locations


def get_polaris_list(usgs_location_filename):
    """
    Loads the list of USGS stations from the usgs location .csv file

    :param usgs_location_filename:
    :return:
    """
    csv_file = open(usgs_location_filename, 'r')
    locations = {}
    n = 0
    for row in csv_file:
        if n == 0:
            n = 1
            continue
        r = row.split(',')
        station_id = r[0]
        latitude = float(r[1]) + float(r[2]) / 60.
        longitude = -(float(r[3]) + float(r[4]) / 60.)
        locations[station_id] = {'long': longitude, 'lat': latitude}
    return locations


def search_nc_file(nc_file):
    print nc_file
    nc_data = nc.Dataset(nc_file)
    nc_time = nc_data.getncattr('start_date')
    t1 = nc_time.split(".")
    n, m = nc_data.variables['lat'].shape
    distances = locations.copy()
    for k in distances.keys():
        distances[k]['dist'] = np.inf
    for i in range(0, n):
        for j in range(0, m):
            _lat = nc_data.variables['lat'][i, j]
            _long = nc_data.variables['lon'][i, j]
            for k in distances.keys():
                _dist = (_lat - distances[k]['lat'])**2 + (_long - distances[k]['long'])**2
                if distances[k]['dist'] > _dist:
                    distances[k]['dist'] = _dist
                    distances[k]['index'] = (i, j)

    x = nc_data.variables['lon'][:, :]
    y = nc_data.variables['lat'][:, :]
    z = nc_data.variables['turbidity'][:, :]

    cc_data = {}
    for k in distances.keys():
        var_data = {'cc_lat': y[distances[k]['index']], 'cc_long': x[distances[k]['index']],
                    'insitu_lat':  distances[k]['lat'], 'insitu_long':  distances[k]['long'],
                    'distance': distances[k]['dist']}

        v_i, v_j = distances[k]['index']
        for vn in cc_names:
            var_data[vn] = nc_data.variables[vn][v_i, v_j]
        cc_data[k] = var_data
    return t1[0], cc_data


def load_variable_names(filename):
    f = open(filename, 'r')
    varnames = []
    for line in f:
        l = line.split()
        if l[0] == '#':
            continue
        varnames.append(l[0])
    return varnames


def parallel_filter_coastcolour_data():
    data_dir = '/Users/jadelson/Documents/phdResearch/SedimentLearning/2015_Project/data/'
    global cc_names
    cc_names = load_variable_names(data_dir + 'coastcolour/coastcolour_types.csv')
    usgs_filename = data_dir + 'usgs/usgs_locations.csv'
    polaris_filename = data_dir + 'polaris/polaris_location.csv'
    usgs_locations = get_usgs_list(usgs_filename)
    polaris_locations = get_polaris_list(polaris_filename)

    global locations
    for l in usgs_locations.keys():
        locations[l] = usgs_locations[l]
    for l in polaris_locations.keys():
        locations[l] = polaris_locations[l]

    cc_data = {}
    for l in locations:
        cc_data[l] = {}
    t0 = time.time()

    # cc_time, _cc_data = search_nc_file(locations, cc_names, data_dir + 'coastcolour/productions_may2015/
    # MER_FSG_CCL2W_20040115_184309_000001742023_00242_09815_8723.nc')

    inputs = []
    for f in os.listdir(data_dir + 'coastcolour/productions_may2015'):
        if f.endswith(".nc"):
            inputs.append(data_dir + 'coastcolour/productions_may2015/' + f)

    pool = mp.Pool(processes=7)

    t1 = time.time()

    result = dict(pool.map(search_nc_file, inputs))

    total_time = t1-t0

    for k in result.keys():
        for l in cc_data.keys():
            cc_data[l][k] = result[k][l]

    print total_time

    filebase = data_dir + 'coastcolour/test_data/coastcolour_data_'

    for l in cc_data.keys():
        with open(filebase + l + '.csv', 'w') as f:
            w = csv.writer(f)
            for k in cc_data[l].keys():
                w.writerow(['date_time'] + cc_data[l][k].keys())
                break
            f.close()

    for l in cc_data.keys():
        loc_data = cc_data[l]
        with open(filebase + l + '.csv', 'a') as f:
            w = csv.writer(f)
            for k in cc_data[l].keys():

                w.writerow([k] + cc_data[l][k].values())
            f.close()


def change_polaris(locations, filename, filename_base):
    """
    From the big ugly file you download from polaris this seperates the data to the .csv group format
    :param locations:
    :param filename:
    :param filename_base:
    """

    data = {}
    names = []
    for l in locations:
        data[l] = []
    with open(filename, 'r') as f:
        myreader = csv.reader(f)
        names = myreader.next()
        units = myreader.next()

        for row in myreader:
            d = {}
            for i in range(0, len(row)):
                if i == 0:
                    d[names[0] + names[1]] = row[0] + ' ' + row[1]
                elif i == 1:
                    continue
                else:
                    d[names[i]] = row[i]
            data[str(int(round(float(row[2]))))].append(d)

    for l in data.keys():
        if len(data[l]) > 0:
            with open(filename_base + 'polaris_data_' + l + '.csv', 'w') as f:
                mywriter = csv.writer(f)
                name_row = []
                for i in range(0, len(names)):
                    if i == 0:
                        name_row.append(names[0] + names[1])
                    elif i == 1:
                        continue
                    else:
                        name_row.append(names[i])
                mywriter.writerow(name_row)

                for line in data[l]:
                    row = []
                    for n in name_row:
                        row.append(line[n])

                    mywriter.writerow(row)





if __name__ == '__main__':
    data_dir = '../data/'

    # parallel_filter_coastcolour_data()
    polaris_locations = get_polaris_list(data_dir + 'polaris/polaris_locations.csv')
    usgs_locations = get_usgs_list(data_dir + 'usgs/usgs_locations.csv')
    start_date = '2002-01-01'
    end_date = '2012-12-31'
    # for l in usgs_locations:
    #     download_usgs_data(l, start_date, end_date, data_dir + 'usgs/')

    change_polaris(polaris_locations, data_dir + 'polaris/all_polaris_data.csv', data_dir + 'polaris/')