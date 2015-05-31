import datetime

__author__ = 'jadelson'
import get_data
import numpy as np
import pandas as pd

data_dir = '/Users/jadelson/Documents/phdResearch/SedimentLearning/2015_Project/data/'


def load_pandas(locations, filename_base):
    """
    Loads the .csv files into a dictionary of pandas dataframes
    :param locations:
    :param filename_base:
    :return:
    """
    cc_dataframes = {}
    for l in locations:
        df = pd.DataFrame.from_csv(filename_base + l + '.csv')
        df = df.sort()
        cc_dataframes[l] = df
    return cc_dataframes


def final_csv(locations, cc_data, other_data, base):
    for l in locations:
        a = cc_data[l]
        b = other_data[l]
        # print b
        dothis = 0

        for i in a.index:
            time_diff = i - b.index
            j = np.argmin(np.abs(time_diff))
            s = b.iloc[j]
            if dothis == 0:
                for k in s.keys():
                    a[k] = np.nan
                a['time_diff'] = np.nan
                a[base + '_time'] = np.nan
                dothis = 1

            for k in s.keys():
                a.loc[i, k] = s[k]
            a.loc[i, 'time_diff'] = time_diff[j]
            a.loc[i, base + '_time'] = b.index[j]
        a = a[a.l2r_flags & 2048 != 2048]
        filter_time = 0.3
        a = a[a.time_diff < datetime.timedelta(days=filter_time)]
        a = a[a.time_diff > datetime.timedelta(days=-filter_time)]
        a.to_csv('../data/full/' + base + l + '.csv')


def load_full_data():
    data_dir = '../data/'
    usgs_filename = data_dir + 'usgs/usgs_locations.csv'
    polaris_filename = data_dir + 'polaris/polaris_locations.csv'
    usgs_locations = get_data.get_usgs_list(usgs_filename)
    polaris_locations = get_data.get_polaris_list(polaris_filename)
    locations = {}
    for l in usgs_locations.keys():
        locations[l] = usgs_locations[l]
    for l in polaris_locations.keys():
        locations[l] = polaris_locations[l]
    cc_data = load_pandas(locations, data_dir + 'coastcolour/test_data/coastcolour_data_')
    tz_choice = 'UTC'
    for l in cc_data.keys():
        cc_data[l] = cc_data[l].tz_localize('UTC')
        cc_data[l] = cc_data[l].tz_convert(tz_choice)
    usgs_data = load_pandas(usgs_locations, data_dir + 'usgs/usgs_data_')
    for l in usgs_data.keys():
        usgs_data[l] = usgs_data[l].tz_localize('Etc/GMT-7')
        usgs_data[l] = usgs_data[l].tz_convert(tz_choice)
    polaris_data = load_pandas(polaris_locations, data_dir + 'polaris/polaris_data_')
    for l in polaris_data.keys():
        polaris_data[l] = polaris_data[l].tz_localize('US/Pacific')
        polaris_data[l] = polaris_data[l].tz_convert(tz_choice)

    final_csv(usgs_locations, cc_data, usgs_data, 'usgs')
    final_csv(polaris_locations, cc_data, polaris_data, 'polaris')


if __name__ == '__main__':
    load_full_data()


def blah():
    data_dir = '../data/'
    usgs_filename = data_dir + 'usgs/usgs_locations.csv'
    polaris_filename = data_dir + 'polaris/polaris_locations.csv'
    usgs_locations = get_data.get_usgs_list(usgs_filename)
    polaris_locations = get_data.get_polaris_list(polaris_filename)

    locations = {}
    for l in usgs_locations.keys():
        locations[l] = usgs_locations[l]
    for l in polaris_locations.keys():
        locations[l] = polaris_locations[l]

    cc_data = load_pandas(locations, data_dir + 'coastcolour/test_data/coastcolour_data_' )


    return cc_data