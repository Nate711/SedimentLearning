__author__ = 'jadelson'
# Code for checking the sediment values against in situ data. Before the regression is run csv files of data must be
# created. *_63680 codes for turbidty, *_99409 codes for suspended sediment concetration

from sklearn import linear_model, cross_validation, decomposition
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import csv

# All possible inputs (Some are integer flags that should not be used)
'''
x_names = ['Kd_412', 'Kd_443', 'Kd_490', 'Kd_510', 'Kd_560', 'Kd_620', 'Kd_664', 'Kd_680', 'Kd_709', 'Kd_754', 'Kd_min',
           'Z90_max', 'conc_chl_merged', 'conc_chl_nn', 'conc_chl_oc4', 'conc_chl_weight', 'conc_tsm', 'iop_a_det_443',
           'iop_a_dg_443', 'iop_a_pig_443', 'iop_a_total_443', 'iop_a_ys_443', 'iop_b_tsm_443', 'iop_b_whit_443',
           'iop_bb_spm_443', 'iop_quality', 'owt_class_1', 'owt_class_2', 'owt_class_3', 'owt_class_4', 'owt_class_5',
           'owt_class_6', 'owt_class_7', 'owt_class_8', 'owt_class_9', 'owt_class_sum', 'reflec_1', 'reflec_10',
           'reflec_12', 'reflec_13', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_6', 'reflec_7', 'reflec_8',
           'reflec_9', 'turbidity']
'''

# Basic Reflectance names
x_names = ['reflec_1', 'reflec_10',
           'reflec_12', 'reflec_13', 'reflec_2', 'reflec_3', 'reflec_4', 'reflec_5', 'reflec_6', 'reflec_7', 'reflec_8',
           'reflec_9']

# y values read from usgs online data
# y_names = ['04_80154', '05_80154', '06_63680', '07_63680', '03_63680', '09_80154', '10_80154','Calculated SPM']

y_names = ['05_80154']
# USGS data code for this particular y value
Y_CODE = '05_80154'


def ridge_regression(x_train, x_test, y_train, y_test, save_name, this_alpha=0, title=''):
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
    clf = linear_model.Ridge(alpha=this_alpha)
    clf.fit(x_train, y_train)
    print clf
    print x_test.shape
    y_pred = clf.predict(x_test)

    plt.plot(y_pred, y_test, 'ok')
    plt.title(r'%s Ridge Regression $\alpha$ = %s' % (title, this_alpha))
    plt.xlabel('Predicted turbidity')
    plt.ylabel('In situ measure turbidity')
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
    kf = cross_validation.KFold(m, n_folds=folds, random_state=4)
    errors = np.zeros(5)
    x_data = np.array(x_data1)
    y_data = np.array(y_data1)
    i = 0
    for train_index, test_index in kf:
        x_train, x_test = x_data[train_index, :], x_data[test_index, :]
        y_train, y_test = y_data[train_index], y_data[test_index]
        test_error, train_error, y_pred, y_train_pred = ridge_regression(x_train, x_test, y_train, y_test,
                                                                         'all_remote_data_ridge2.png', param,
                                                                         'Raw in situ Data, no tide information.')

        errors[i] = test_error
        i += 1
    return np.mean(errors)


def get_data(filenames):
    """
    Read in data from csv files.

    Nathan-
    Grabs all reflectance values from csv by checking column against x_names.

    Grabs SPM values from usgs and polaris csvs.

    :param filenames: datafile
    :return: feature inputs (row vector, i think), feature outputs (column vector)
    """

    # np.array is a function!!!!

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
                # control flag: good when no empty data, not good when empty data
                good = False

                y = np.nan

                # iterate through csv keys including reflect, 08_blah, etc
                for n in row.keys():

                    # if the key is a turbidity measurement
                    if n in y_names:
                        if not row[n] == '':
                            y = float(row[n])

                            good = True
                            continue

                # only add values to dicts if no missing data
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


def find_best_shrink_polynomial_degree_ridgee(x_data, y_data, save_flag):
    """
    Does a parameter sweep and uses kmeans cross validation to find the optimal ridge parameter

    :param x_data: feature inputs
    :param y_data: feature outputs
    :param save_flag: flag to save file
    """
    step = 10
    # log_alphas = [range(-100, 60, step), range(40, 110, step), range(100, 150, step), range(210, 230, step)]
    log_alphas = [range(-100, 200, step), range(-10, 200, step)]

    degrees = [1, 2]
    for (degree, log_alpha) in zip(degrees, log_alphas):
        print degree
        # for degree in [5,6]:
        #     print degree
        #     log_alpha = range(-100, 1000, step)
        # if degree < 7:
        #     continue
        alpha = []
        errors = []
        for a in log_alpha:
            a = 2.0 ** (a / 10.0)
            try:
                x = np.sqrt(kfolds_ridge(x_data, y_data, a))
                # x = np.sqrt(kfolds_poly_ridge(x_data, y_data, degree, a))
                if x < 100000:
                    errors.append(x)
                    alpha.append(a)
            except ValueError:
                pass
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
    if save_flag:
        plt.savefig('/Users/Nathan/Desktop/Turbidity/SedimentLearning/figures/polynomial_ridge')
        # plt.savefig('/Users/jadelson/Documents/phdResearch/SedimentLearning/figures/polynomial_ridge')


def main():
    """
    Main function, must call get data then do some regression work
    """

    # mypath = '/Users/jadelson/Dropbox/SedimentLearning/data/full/'

    mypath = '/Users/Nathan/Dropbox/SedimentLearning/data/full/'

    from os import listdir
    from os.path import isfile, join

    # grabs all the filenames of csvs inside data/full/
    # polaris11.csv, usgs...csv, etc
    filenames = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.csv')]

    X, y = get_data(filenames)

    # X, y = get_data(['/Users/jadelson/Dropbox/SedimentLearning/data/full/polaris8.csv'])
    # X, y = get_data(['/Users/Nathan/Dropbox/SedimentLearning/data/full/polaris8.csv'])


    print kfolds_ridge(X, y, 0.5)


# find_best_shrink_polynomial_degree_ridgee(X,y,True)
# print X.shape, y.shape


# why is the shape of X blah,blah but the shape of y blah, even though they're both 2d????????

if __name__ == '__main__':
    main()
