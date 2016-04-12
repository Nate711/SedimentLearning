__author__ = 'jadelson'
# Code for checking the sediment values against in situ data. Before the regression is run csv files of data must be
# created. *_63680 codes for turbidty, *_99409 codes for suspended sediment concetration

import csv
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation, linear_model, svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import robust_scale
from itertools import combinations, permutations

# All possible inputs (Some are integer flags that should not be used)
'''
x_names = ['Kd_412', 'Kd_443', 'Kd_490', 'Kd_510', 'Kd_560', 'Kd_620', 'Kd_664', 'Kd_680', 'Kd_709', 'Kd_754', 'Kd_min',
           'Z90_max', 'conc_chl_merged', 'conc_chl_nn', 'conc_chl_oc4', 'conc_chl_weight', 'conc_tsm', 'iop_a_det_443',
           'iop_a_dg_443', 'iop_a_pig_443', 'iop_a_total_443', 'iop_a_ys_443', 'iop_b_tsm_443', 'iop_b_whit_443',
           'iop_bb_spm_443', 'iop_quality', 'owt_class_1', 'owt_class_2', 'owt_class_3', 'owt_class_4', 'owt_class_5',
           'owt_class_6', 'owt_class_7', 'owt_class_8', 'owt_class_9', 'owt_class_sum', 'reflec_1', 'reflec_10',
           'reflec_12', 'reflec_13', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_6', 'reflec_7', 'reflec_8',
           'reflec_9', 'turbidity']

# Basic Reflectance names
x_names = ['reflec_1', 'reflec_10',
           'reflec_12', 'reflec_13', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_6', 'reflec_7', 'reflec_8',
           'reflec_9']
x_names = ['reflec_1', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5','reflec_7']

# y values read from usgs online data
# y_names = ['04_80154', '05_80154', '06_63680', '07_63680', '03_63680', '09_80154', '10_80154','Calculated SPM']

y_names = ['Calculated SPM']
#y_names = ['Calculated Chlorophyll']

# USGS data code for this particular y value
#Y_CODE = '05_80154'

Y_CODE = 'Calculated SPM'
#Y_CODE = 'Calculated Chlorophyll'

'''

'''
Notes:
Source: http://waterdata.usgs.gov/ca/nwis/dv?format=rdb&site_no=11525535&referred_module=sw&begin_date=1900-1-1&end_date=2008-12-31
DD: 05 Statistic: 80154  Parameter:00003 is the: Suspended sediment concentration, milligrams per liter (Mean)
Source: http://sfbay.wr.usgs.gov/access/wqdata/query/qhelp.html
CALCULATED SPM: estimated concentration of suspended sediments, calculated from the OBS voltage output and
linear regression (calibration) between the discrete measures of suspended solids and the OBS voltage.
The standard error of the calculated value for each cruise is listed at the top of the data table.
Units of measurement are milligrams per liter.

'''


def ridge_regression(x_train, x_test, y_train, y_test, save_name=np.nan, this_alpha=0, title=''):
    # print 'Ridge regression alpha =', this_alpha
    """
    ridge_regression fits a linear model with norm 2 regularization term for the ridge regression

    :param x_train: set of training feature inputs
    :param x_test: set of testing feature inputs
    :param y_train: set of training feature outputs
    :param y_test: set of testing feature outputs
    :param save_name: name of image file if plotting
    :param this_alpha: ridge (regularization) parameter
    :param title: title of plot
    :return: mean squared error of test data, mean square error of training data, predicted test set outputs, predicted training set outputs
    """

    # fits the linear ridge model, scikit makes it all easy
    clf = linear_model.Ridge(alpha=this_alpha)
    clf.fit(x_train, y_train)
    # print clf.coef_

    # print clf
    # print x_test.shape
    y_pred = clf.predict(x_test)

    plt.plot(y_pred, y_test, 'ok')
    plt.title(r'%s Ridge Regression $\alpha$ = %s' % (title, this_alpha))
    plt.xlabel('Predicted turbidity')
    plt.ylabel('In situ measure turbidity')

    # is this a valid way of checking for a save? should use a flag?
    if save_name is not np.nan:
        plt.savefig('/Users/Nathan/Desktop/Turbidity/SedimentLearning/figures/' + save_name)
        # plt.show()

    return mean_squared_error(y_pred.tolist(), y_test.tolist()), \
           mean_squared_error(y_train.tolist(), clf.predict(x_train).tolist()), \
           y_pred, clf.predict(x_train)


def kfolds_ridge(x_data1, y_data1, param):
    """
    kfolds_ridge does a k folds cross validation on the dataset, seperating the data into some number of folds (5) of
    test sets. Designed for ridge regression but could easily be modified for other regression analysis/

    :param x_data1: input data
    :param y_data1: output data
    :param param: parameter for regression function (alpha in ridge regression)
    :return: mean error
    """
    m = len(y_data1)
    folds = 5

    # get indices for kfolds validation
    kf = cross_validation.KFold(m, n_folds=folds, random_state=4)

    errors = np.zeros(5)
    x_data = np.array(x_data1)
    y_data = np.array(y_data1)

    # what's the point of i?
    i = 0
    for train_index, test_index in kf:
        x_train, x_test = x_data[train_index, :], x_data[test_index, :]
        y_train, y_test = y_data[train_index], y_data[test_index]

        # run the ridge regression
        test_error, train_error, y_pred, y_train_pred = ridge_regression(x_train, x_test, y_train, y_test,
                                                                         'all_remote_data_ridge2.png', param,
                                                                         'Raw in situ Data, no tide information.')

        errors[i] = test_error
        i += 1
    return np.mean(errors)


def get_data(x_names=['reflec_1', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_7'], y_names=['Calculated SPM'], filenames=['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_8hr.csv'], Y_CODE='Calculated SPM'):
    # TODO use pandas csv reader methods instead of for loop weirdness
    """
    Read in data from csv files.

    Grabs all reflectance values from csv by checking column against x_names.
    Grabs SPM values

    :param filenames: datafile
    :return: feature inputs (row vector, i think), feature outputs (column vector)
    """

    # initialize dictionaries to empty arrays
    x_dict = {}
    y_dict = {}
    for n in x_names:
        x_dict[n] = np.zeros((0, 0))
    for n in y_names:
        y_dict[n] = np.zeros((0, 0))

    for f in filenames:
        with open(f, 'rb') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')

            for row in reader:
                # control flag: good when no empty data in row, not good when empty data in row
                good = False

                # default y value, could be a problem / introduce bugs
                y = np.nan

                # iterate through csv keys including reflect, 08_blah, etc
                for n in row.keys():

                    # potential error: assumes only one y_name we want
                    # check if SPM is in the row
                    if n in y_names:
                        if not row[n] == '':
                            y = float(row[n])

                            good = True
                            continue
                    # check if there are empty x values, set to bad if there are
                    if n in x_names:
                        if row[n] == '':
                            good = False
                            print 'bad data'

                            # break out of for loop b/c don't care if there are other empties
                            continue

                # only add values to dicts if no missing data, again, assumes only one y
                for n in row.keys():
                    if good:
                        if n in x_names:
                            x_dict[n] = np.append(x_dict[n], float(row[n]))
                        if n in y_names:
                            y_dict[n] = np.append(y_dict[n], y)

    '''
    for n in x_names:
        print (x_dict[n]).shape
    for n in y_names:
        print (y_dict[n]).shape
    '''

    X = np.array(x_dict.values())

    # The shape of y is (7,) because it's non-rectangular, each row in y has a different length
    # Each row of x is the same length so it reads (3003,12)

    Y = np.array(y_dict[Y_CODE])

    # print X.shape,Y.shape
    # print Y

    return X.transpose(), Y


def find_best_shrink_polynomial_degree_ridgee(x_data, y_data, save_flag, name_tag=''):
    """
    Does a parameter sweep and uses kmeans cross validation to find the optimal ridge parameter. Currently only linear

    :param x_data: feature inputs
    :param y_data: feature outputs
    :param save_flag: flag to save file
    """
    # sweep through powers of 2
    step = 1
    # log_alphas = [range(-100, 60, step), range(40, 110, step), range(100, 150, step), range(210, 230, step)]
    log_alphas = [range(-10, 20, step), range(-1, 20, step)]

    # ridge regression only implemented for linear right now
    # degrees = [1,2]
    degrees = [1]

    for (degree, log_alpha) in zip(degrees, log_alphas):
        # print degree
        # for degree in [5,6]:
        #     print degree
        #     log_alpha = range(-100, 1000, step)
        # if degree < 7:
        #     continue
        alpha = []
        errors = []
        for a in log_alpha:
            a = 2.0 ** (a)
            try:
                # sqrt of mean squared error
                x = np.sqrt(kfolds_ridge(x_data, y_data, a))

                # x = np.sqrt(kfolds_poly_ridge(x_data, y_data, degree, a))
                if x < 100000:
                    errors.append(x)
                    alpha.append(a)
            except ValueError:
                pass
        # skip to the next degree w/o printing if no good alphas found
        if len(alpha) == 0:
            continue
        print 'OUTPUT: RIDGE DEGREE:', degree
        print 'OUTPUT: RIDGE PARAM:', alpha[np.argmin(errors)]
        print 'OUTPUT: RIDGE RME:', np.min(errors)
        degrees.append('Polynomial degree: ' + str(degree))

        plt.semilogx(alpha, errors)
    plt.title(r'Cross Validation for Shrinkage Parameters (Polynomial Regression)')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'K-Folds averaged Root Mean Square Error')
    plt.legend(degrees, loc='upper left')
    # plt.show()
    if save_flag:
        plt.savefig('/Users/Nathan/Desktop/Turbidity/SedimentLearning/figures/polynomial_ridge' + '_' + name_tag)
        plt.show()
        # plt.savefig('/Users/jadelson/Documents/phdResearch/SedimentLearning/figures/polynomial_ridge')


def simple_linearSVR(X, y):
    '''
    Perform a simple svr on the data.
    RMSE for usgs/coastcolor is 70.6
    RMSE for landsat/polaris (2hr) is 105.2
    :param X:
    :param y:
    :return:
    '''
    clf = svm.LinearSVR(epsilon=0.0)
    clf.fit(X, y)

    y_predict = clf.predict(X)

    plt.plot(y_predict, y, '.k')
    plt.title('Actual SPM vs Predicted SPM (SVR)')
    plt.xlabel('Predicted SPM (mg/L)')
    plt.ylabel('Actual SPM (mg/L)')
    print 'SVR OUTPUT: ROOT MEAN SQUARED ERROR: ' + str(np.sqrt(mean_squared_error(y_predict.tolist(), y.tolist())))
    plt.show()


def simple_ridgeCV(X, y):
    '''
    Perform a ridge regression with cross validation.
    RMSE for usgs/coastcolor is 57.2
    RMSE for landsat/polaris (2hr) is 11.8
    RMSE for landsat/polaris (4hr) is 19.0
    RMSE for landsat/polaris (8hr) is 34.4
    RMSE for landsat/polaris (12hr) is 34.4 - same data as 8hr
    RMSE for landsat/polaris (16hr) is 34.4 - same data as 12hr
    RMSE for landsat/polaris (20hr) is 33.4
    RMSE for landsat/polaris (24hr) is 28.6

    :param X: input data
    :param y: output
    :return: nothing
    '''
    log_alphas = np.array(np.arange(-15, 15, 0.25), dtype='float64')
    alphas = 2 ** log_alphas
    clf = linear_model.RidgeCV(alphas=alphas, cv=None, store_cv_values=True)

    clf.fit(X, y)

    print 'OUTPUT: ALPHA: ' + str(clf.alpha_)

    y_predict = clf.predict(X)
    print 'Coefficients (index, coeff): ' + str(np.array(zip(np.arange(X.shape[0]), clf.coef_)))

    graph_actual_SPM_vs_predicted_SPM(y, y_predict)
    return clf


def division_feature_expansion(X):
    '''
    index = first index*(num indices) + second index - 1
    :param X: input matrix with each datum as a row
    :return: new matrix with each column as the ratio between two of the original feature arrays
             the returned matrix does not contain the original columns
    '''
    xt = X.T

    x_new = np.array([]).reshape(0, X.shape[0])

    indices = np.arange(X.shape[1])
    for (a, b) in permutations(indices, 2):
        offset = .001  # avoid divide by zero errors??? totally arbitrary

        x_new = np.append(x_new, np.array([(xt[a] + offset) / (xt[b] + offset)]), axis=0)
        # print x_new.shape[0] - 1, (a, b)

    # print x_new.shape
    return x_new.T


def Han_EA_MB(Rrs_lambda1, Rrs_lambda2, Rrs_lambda3, spm):
    # log10(SPM) = c0 + c1*x1 + c2*x2
    # x1 = Rrs(lambda1) + Rrs(lambda2) sensitive to spm
    # x2 = Rrs(lambda3)/Rrs(lambda1) compensating term
    # lambda 1 = green (555) = band2 for landsat 457, lambda 2 = red (670) = band3 for landsat 457,
    # lanbda3 = blue (490) = band1 for landsat 457

    logy = np.log10(spm)

    X1 = Rrs_lambda1 + Rrs_lambda2
    X2 = np.divide(Rrs_lambda3, Rrs_lambda1)
    C0 = np.ones(Rrs_lambda1.size)

    new_X = np.array([C0, X1, X2]).T

    # start ridge CV for log data
    log_alphas = np.array(np.arange(-15, 15, 0.25), dtype='float64')
    alphas = 2 ** log_alphas
    clf = linear_model.RidgeCV(alphas=alphas, cv=None, store_cv_values=True)
    clf.fit(new_X, logy)

    y_predict = clf.predict(new_X)
    print('EA-MB log regression')
    graph_actual_SPM_vs_predicted_SPM(logy, y_predict)

    y_predict_normal = np.power(10, y_predict)

    print('EA-MB regression')
    graph_actual_SPM_vs_predicted_SPM(spm, y_predict_normal)


def Han_EA_MB_LANDSAT():
    # log10(SPM) = c0 + c1*x1 + c2*x2
    # x1 = Rrs(lambda1) + Rrs(lambda2) sensitive to spm
    # x2 = Rrs(lambda3)/Rrs(lambda1) compensating term
    # lambda 1 = green (555) = band2 for landsat 457, lambda 2 = red (670) = band3 for landsat 457,
    # lanbda3 = blue (490) = band1 for landsat 457
    x_names = ['reflec_1', 'reflec_2', 'reflec_3']
    y_names = ['Calculated SPM']

    filenames = ['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_8hr.csv']

    X, y = get_data(x_names=x_names, y_names=y_names, filenames=filenames, Y_CODE='Calculated SPM')
    # X col 0 is band1, col 1 is band2, col 2 is band3
    # so lambda1 is col 1, lambda2 is col 2, lambda3 is col 0

    Han_EA_MB(X[:, 1], X[:, 2], X[:, 0], y)


def Han_EA_MB_MERIS():
    # log10(SPM) = c0 + c1*x1 + c2*x2
    # x1 = Rrs(lambda1) + Rrs(lambda2) sensitive to spm
    # x2 = Rrs(lambda3)/Rrs(lambda1) compensating term
    # lambda 1 = green (555) = band2 for landsat 457, lambda 2 = red (670) = band3 for landsat 457,
    # lanbda3 = blue (490) = band1 for landsat 457
    x_names = ['reflec_3', 'reflec_5', 'reflec_7']
    y_names = ['Calculated SPM']

    mypath = '/Users/Nathan/Dropbox/SedimentLearning/data/full/'
    filenames = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.csv')]
    # filenames = ['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_8hr.csv']

    X, y = get_data(x_names=x_names, y_names=y_names, filenames=filenames, Y_CODE='Calculated SPM')

    Han_EA_MB(Rrs_lambda1=X[:, 1], Rrs_lambda2=X[:, 2], Rrs_lambda3=X[:, 0], spm=y)


def Han_EA_BR():
    '''
    spm = a0 * exp(a1*X)
    X1 = Rrs(lambda1)/Rrs(lambda2)
    lambda1 is near IR 865nm -> band 4
    lambda2 is green 555nm -> band 2

    do log(spm) = log(a0) + a1*X?
    :return:
    '''
    x_names = ['reflec_2', 'reflec_4']
    # col 0 is lambda2, col 1 is lambda 1
    y_names = ['Calculated SPM']

    filenames = ['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_8hr.csv']
    X, y = get_data(x_names=x_names, y_names=y_names, filenames=filenames, Y_CODE=y_names[0])

    X1 = np.divide(X[:, 1], X[:, 0])

    logy = np.log10(y)
    A0 = np.ones(logy.shape)

    new_X = np.array([A0, X1]).T
    print new_X
    print logy

    log_alphas = np.array(np.arange(-15, 15, 0.25), dtype='float64')
    alphas = 2 ** log_alphas
    clf = linear_model.RidgeCV(alphas=alphas, cv=None, store_cv_values=True)
    clf.fit(new_X, logy)

    y_predict = clf.predict(new_X)
    print('EA-BR log regression')
    graph_actual_SPM_vs_predicted_SPM(logy, y_predict)

    y_predict_normal = np.power(10, y_predict)

    print('EA-BR regression')
    graph_actual_SPM_vs_predicted_SPM(y, y_predict_normal)


def graph_actual_SPM_vs_predicted_SPM(actual, predicted):
    plt.plot(predicted, actual, '.k')
    plt.title('Actual SPM vs Predicted SPM (Ridge)')
    plt.xlabel('Predicted SPM (mg/L)')
    plt.ylabel('Actual SPM (mg/L)')
    print 'Ridge Output: RMSE: ' + str(np.sqrt(mean_squared_error(actual, predicted.tolist())))
    print 'Ridge Output: R^2: ' + str(r2_score(actual.tolist(), predicted.tolist()))
    x_line = np.linspace(0, np.amax(predicted), 10)
    y_line = x_line
    plt.plot(x_line, y_line, '-r')
    plt.show()


def empirical_band_ratio():
    '''

    :return:
    '''
    ## DO LANDSAT REGRESSION
    x_names = ['reflec_1', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_7']
    y_names = ['Calculated SPM']

    filenames = ['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_2hr.csv']

    X, y = get_data(x_names=x_names, y_names=y_names, filenames=filenames, Y_CODE='Calculated SPM')
    y1 = np.log(y)

    # make X only ratios between bands
    # X = np.append(X,division_feature_expansion(X),axis=1)
    X = division_feature_expansion(X)
    '''
    Ratios with highest correlation on 8 hour data
    29: 1.91
    9: -1.63
    14: -1.57
    28: 1.33
    5: 1.15
    '''

    # reflec 5 vs 7 and 4 vs 1
    # X = X[:,-1:]
    # X[:,-1] = np.exp(X[:,-1])
    # the last ratio varies more with log y than y

    print 'LANDSAT-POLARIS samples: {}   features: {}'.format(X.shape[0], X.shape[1])

    # start ridge CV for log data

    # array of alphas to test
    log_alphas = np.array(np.arange(-15, 15, 0.25), dtype='float64')
    alphas = 2 ** log_alphas

    # fit ridge with cv model
    clf = linear_model.RidgeCV(alphas=alphas, cv=None, store_cv_values=True)
    clf.fit(X, y1)

    indices_and_coefs = np.array(zip(np.arange(X.shape[1]),clf.coef_))
    print 'Weights of band ratios sorted by correlation (index,weight)'
    print indices_and_coefs[np.argsort(np.abs(clf.coef_))[::-1]]

    y_predict = clf.predict(X)
    print('\nLog(spm) regression on all band ratios')
    graph_actual_SPM_vs_predicted_SPM(y1, y_predict)
    print('exp(log(spm) prediction on all band ratios')
    graph_actual_SPM_vs_predicted_SPM(y, np.exp(y_predict))

    # ONLY top 5 bands
    X = X[:,[29,9,14,28,5]]
    # ONLY top 3 bands
    # X = X[:,[29,9,14]]

    clf2 = linear_model.RidgeCV(alphas=alphas,cv=None,store_cv_values=True)
    clf2.fit(X,y1)
    y_predict2 = clf2.predict(X)
    print('\nLog(spm) regression on top five band ratios')
    graph_actual_SPM_vs_predicted_SPM(y1,y_predict2)
    print('exp(log(spm) prediction on top five band ratios')
    graph_actual_SPM_vs_predicted_SPM(y,np.exp(y_predict2))

def main():
    '''
    Main function, must call get data then do some regression work
    '''

    '''
    ##### do regression with coastcolor and polaris data
    x_names = ['reflec_1', 'reflec_10',
           'reflec_12', 'reflec_13', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_6', 'reflec_7', 'reflec_8',
           'reflec_9']
    y_names = ['Calculated SPM']

    # grabs all the filenames of csvs inside data/full/
    # polaris11.csv, usgs...csv, etc
    # mypath = '/Users/jadelson/Dropbox/SedimentLearning/data/full/'
    mypath = '/Users/Nathan/Dropbox/SedimentLearning/data/full/'
    filenames = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.csv')]

    X, y = get_data(x_names = x_names,y_names=y_names,filenames = filenames,Y_CODE = 'Calculated SPM')

    print 'CC-USGS samples: {}   features: {}'.format(X.shape[0],X.shape[1])

    simple_ridgeCV(X,y)
    #simple_linearSVR(X,y)
    '''


def test():
    x_names = ['reflec_1', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_7']
    y_names = ['Calculated SPM']

    filenames = ['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_8hr.csv']

    X, y = get_data(x_names=x_names, y_names=y_names, filenames=filenames, Y_CODE='Calculated SPM')

    Xs = robust_scale(X, axis=0)
    # print Xs

    # print y
    ys = robust_scale(y.reshape(-1, 1), axis=0)
    # print ys

    simple_ridgeCV(Xs, ys)


if __name__ == '__main__':
    # test()
    # main()
    ## DO LANDSAT REGRESSION

    # EA_MB_LANDSAT()
    # EA_MB_MERIS()
    # EA_BR()
    empirical_band_ratio()

''' NOTES

landsat 8 band | wavelength | landsat 4,5,7 band
    1 | 430-450 | none
    2 | 450-510 blue  | 1
    3 | 530-590 green | 2
    4 | 640-670 red   | 3
    5 | 805-880 NIR   | 4
    6 | 1570-1650 SWIR| 5
    7 | 2110 - 2290 SWIR 2 | 7
    8 | 500-680 panchromatic | 8
    none | IR | 6

Ridge with 8hr landsat polaris data without division features:
R2 = .293

coeffs
[-0.0911004   0.02210368 -0.02898023 -0.03669951 -0.08078736  0.21832104]


Ridge with 8hr landsat polaris data with division features:
R2 = .298

coeffs, divisions feature weights are quite small
[ -9.15334879e-02   2.35789838e-02  -3.01392975e-02  -3.69621953e-02 -8.06454136e-02   2.18712566e-01
 1.84183934e-01   1.06010624e-02
  -3.39367896e-03  -2.31273956e-03  -2.97999485e-03  -2.25243609e-02
  -3.00992840e-03  -2.20057625e-03  -2.43230700e-03  -8.18643396e-04
  -5.68472257e-04  -1.18759833e-04   6.50953814e-04   6.07126177e-04
  -6.31267163e-04]


EA-MB regression
R2 for 8hr data: .196 on log regression, .00489 on actual regrssion

R2 for 2hr data: .142 on log regression, .00186 on actual regression
  '''
