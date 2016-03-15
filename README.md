SedimentLearning
================

CS 229 Project: Understanding Remote Sensing Turbidity Data in San Francisco Bay

Modified by Nathan Kau (Intern)

Data directory:


Data_Formatter directory:
get_data.py:
Downloads data online or loads from local disk various remote sensing data including coast color satellite imagery, USGS
sediment levels, and Polaris sensor data. Nothing to do with landsat imagery.

landsat_get_data.py:
Loads Landsat satellite imagery and parses the reflectance data into a csv with data for all usgs and polaris locations.
This csv will have to be filtered to find the data taken at the same time as the polaris / usgs sensors. The csv is
formatted for for use in machine learning algorithms. The csv is formatted with the following columns

load_data.py:
Loads remote sensing data from csvs and other harddisk data.

station_img_coordinates.csv:
Contains table of data indicating the lat long of the various polaris collection locations and the corresponding pixel
coordinates in the landsat imagery.

Figures directory:
Output graphs from machine learning.

Learning directory:
Machine learning algorithms to create regression between in-situ data and satellite reflectance.

