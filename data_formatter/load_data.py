__author__ = 'jadelson'
import get_data
import csv
import pandas as pd

data_dir = '/Users/jadelson/Documents/phdResearch/SedimentLearning/2015_Project/data/'


def load_pandas(locations, filebase):
    cc_dataframes = {}
    for l in locations:
        df = pd.DataFrame.from_csv(filebase + l + '.csv')
        df = df.sort()
        cc_dataframes[l] = df
    return cc_dataframes


if __name__ == '__main__':

    data_dir = '/Users/jadelson/Documents/phdResearch/SedimentLearning/2015_Project/data/'
    usgs_filename = data_dir + 'usgs/usgs_locations.csv'
    polaris_filename = data_dir + 'polaris/polaris_locations.csv'
    usgs_locations = get_data.get_usgs_list(usgs_filename)
    polaris_locations = get_data.get_polaris_list(polaris_filename)

    locations = {}
    for l in usgs_locations.keys():
        locations[l] = usgs_locations[l]
    for l in polaris_locations.keys():
        locations[l] = polaris_locations[l]

    # cc_data = load_pandas(locations, data_dir + 'coastcolour/test_data/coastcolour_data_' )

    usgs_data = load_pandas(usgs_locations, data_dir + 'usgs/usgs_data_')

    polaris_data = load_pandas(polaris_locations, data_dir + 'polaris/polaris_data_')