SedimentLearning
================

**CS 229 Project: Understanding Remote Sensing Turbidity Data in San Francisco Bay**

*Modified by Nathan Kau (Intern)*

**Data directory:**
*landsat_data_UTC.csv*
CSV containing all landsat data only matched to polaris data locations, not polaris data collection times. The column names are based off Joe's column names and are as follows:
cf_mask_quality: quality of cloud mask
cloud: from cf_mask tag, is the cloud mask
landsat_scene: name of the landsat scene
lat: corresponding lat of pixel
long corresponding long of pixel
reflec_1 through reflect_7: sr_band values extracted from satellite images
station_ID: corresponding polaris location ID
date_time_UTC: Time image was taken in UTC

See: http://landsat.usgs.gov/CDR_LSR.php

**Data_Formatter directory:**
*get_data.py:*
Methods for downloading data online or loads from local disk various remote sensing data including coast color satellite imagery, USGS
sediment levels, and Polaris sensor data. Nothing to do with landsat imagery.

*landsat_get_data.py:*
Contains methods for loading Landsat satellite imagery and parses the reflectance data into a csv called landsat_data_UTC.csv with data for all polaris locations.
This csv will have to be filtered to find the data taken at the same time as the polaris / usgs sensors. The csv is
formatted for for use in machine learning algorithms. Also filters landsat_data_UTC.csv to filter data for only landsat data that matches with polaris data in time and space.

*load_data.py:*
Loads remote sensing data from csvs and other harddisk data.

*station_img_coordinates.csv:*
Contains table of data indicating the lat long of the various polaris collection locations and the corresponding pixel
coordinates in the landsat imagery.

**Figures directory:**
Output graphs from regression.py

**Learning directory:**
*regression.py*
Machine learning algorithms to create regression between in-situ data and satellite reflectance.
