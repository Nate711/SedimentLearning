SedimentLearning
================

**CS 229 Project: Understanding Remote Sensing Turbidity Data in San Francisco Bay**

*Modified by Nathan Kau (Joe's High School Underling :)*

**Info on Landsat and Meris data**
*Landsat bands*
http://landsat.usgs.gov/best_spectral_bands_to_use.php
landsat 8 band | wavelength           | landsat 4,5,7 band
             1 | 430-450              | none
             2 | 450-510 blue         | 1
             3 | 530-590 green        | 2
             4 | 640-670 red          | 3
             5 | 805-880 NIR          | 4
             6 | 1570-1650 SWIR       | 5
             7 | 2110 - 2290 SWIR 2   | 7
             8 | 500-680 panchromatic | 8
          none | IR                   | 6
   
http://landsat.usgs.gov/documents/cdr_sr_product_guide.pdf
^ Explains valid data range

*Meris (Coast color) bands*
http://www.coastcolour.org/publications/Coastcolour-PUG-v2.2.pdf
page 28
Meris band  |  wavelength
         1  |  413
         2  |  443
         3  |  490
         4  |  510
         5  |  560
         6  |  620
         7  |  665
         8  |  681
         9  |  708
         10 |  753
         12 |  778
         13 |  865

**SPM sources**
*USGS*
Source: http://waterdata.usgs.gov/ca/nwis/dv?format=rdb&site_no=11525535&referred_module=sw&begin_date=1900-1-1&end_date=2008-12-31
DD: 05 Statistic: 80154  Parameter:00003 is the: Suspended sediment concentration, milligrams per liter (Mean)

*Polaris*
Source: http://sfbay.wr.usgs.gov/access/wqdata/query/qhelp.html
CALCULATED SPM: estimated concentration of suspended sediments, calculated from the OBS voltage output and
linear regression (calibration) between the discrete measures of suspended solids and the OBS voltage.
The standard error of the calculated value for each cruise is listed at the top of the data table.
Units of measurement are milligrams per liter.

**Data directory:**
*landsat_data_UTC.csv*
CSV containing all landsat data only matched to polaris data locations, not polaris data collection times. The column names are based off Joe's column names and are as follows:<br/>
cf_mask_quality: quality of cloud mask <br/>
cloud: from cf_mask tag, is the cloud mask<br/>
landsat_scene: name of the landsat scene<br/>
lat: corresponding lat of pixel<br/>
long corresponding long of pixel<br/>
reflec_1 through reflect_7: sr_band values extracted from satellite images<br/>
station_ID: corresponding polaris location ID<br/>
date_time_UTC: Time image was taken in UTC<br/>

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
Output graphs from regression.py, mapping.py, make_plots.py

**Learning directory:**
*regression.py*
Machine learning algorithms to create regression between in-situ data and satellite reflectance. Includes Han et al's 
algorithm's, Nathan's multi band + ratio algorithm, and other various linear regressions.

*make_plots*
Includes methods for graphing performance of huber-based models. 

*convex.py*
Code for huber model.

*mapping.py*
Important code for generating SPM maps of the bay using the landsat imagery and huber model with Nathan's bands and 
band ratios
