__author__ = 'jadelson'
import get_data
import csv
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


if __name__ == '__main__':
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

    a = cc_data['4']
    print a

    b = polaris_data['4']
    print b


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