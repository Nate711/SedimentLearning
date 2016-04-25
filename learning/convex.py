from cvxpy import *
import numpy as np
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


#
# def huber(x, m):
# 	if np.abs(x) <= m:
# 		return x*x
# 	else:
# 		return m*(2*np.abs(x) - m)


def kfolds_convex(x_data1, y_data1, param, random_seed=None, savename='all_remote_data_huber.png', title='',
                  choice='huber'):
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
    kf = cross_validation.KFold(m, n_folds=folds, shuffle=True, random_state=random_seed)
    errors = np.zeros(5)
    x_data = np.array(x_data1)
    y_data = np.array(y_data1)
    i = 0
    for train_index, test_index in kf:
        x_train, x_test = x_data[train_index, :], x_data[test_index, :]
        y_train, y_test = y_data[train_index], y_data[test_index]
        test_error, train_error, y_pred, y_train_pred,theta = robust_test(x_train, y_train, x_test, y_test, param,
                                                             savename, title, choice)

        errors[i] = test_error
        i += 1

    # only returns training,test,pred,etc data on last fold
    return {'testmse': np.mean(errors), 'testr2': r2_score(y_test, y_pred), 'trainr2': r2_score(y_train, y_train_pred),
            'theta':theta,
            'data': {'x_train': x_train, 'y_train': y_train, 'y_train_pred': y_train_pred, 'x_test': x_test,
                     'y_test': y_test, 'y_pred': y_pred}}


def robust_test(x_train, y_train, x_test, y_test, param, save_name, title='', choice='huber'):
    # print 'Ridge regression alpha =', this_alpha
    """
    ridge_regression fits a linear model with norm 2 regularization term for the ridge regression

    :param x_train: set of training feature inputs
    :param x_test: set of testing feature inputs
    :param y_train: set of training feature outputs
    :param y_test: set of testing feature outputs
    :param save_name: name of image file if plotting
    :param param: ridge (regularization) parameter
    :param title: title of plot
    :return: mean squared error of test data, mean square error of training data, predicted test set outputs, predicted training set outputs
    """
    if (choice == 'huber'):
        theta = robust_regression(x_train, y_train, param)
    elif (choice == 'ninf'):
        theta = inf_regression(x_train, y_train)
    # print x_test.shape
    y_pred = x_test.dot(theta)
    y_train_pred = x_train.dot(theta)
    plt.clf()
    plt.plot(y_pred, y_test, 'ob')
    plt.plot(np.arange(0, 1.05 * np.max(y_test), .1), np.arange(0, 1.05 * np.max(y_test), .1), '-k')
    if title == '':
        title = r'%s Robust Regression $\alpha$ = %s' % (title, param)
    plt.title(title)
    plt.xlabel('Remotely Sensed SPM (g/ml)')
    plt.ylabel('In situ measure SPM (g/ml)')
    plt.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)
    plt.savefig(save_name)

    return mean_squared_error(y_pred.tolist(), y_test.tolist()), \
           mean_squared_error(y_train.tolist(), y_train_pred.tolist()), y_pred, y_train_pred,\
           theta


def optimal_parameters(x, y):
    print y
    alpha = Variable()
    beta = Variable()
    gamma = Variable()
    constraint = [gamma < 0]
    obj = Minimize(sum(gamma * log(x) - log(y)))

    prob = Problem(obj)
    prob.solve(solver=ECOS_BB)
    return alpha, beta, gamma


def robust_regression(x, y, delta=1):
    (m, n) = x.shape
    theta = Variable(n)
    # constraint = []
    constraint = [x * theta >= 0]

    # Form objective.
    obj = Minimize(sum(huber(x * theta - y, delta)))

    # Form and solve problem.
    prob = Problem(obj, constraints=constraint)
    prob.solve(solver=ECOS_BB)  # Returns the optimal value.
    # print "status:", prob.status
    # print "optimal value", prob.value
    # print "optimal var", theta.value
    # print "param", delta
    return theta.value


def inf_regression(x, y):
    (m, n) = x.shape
    theta = Variable(n)
    constraint = []
    constraint = [x * theta >= 0]

    # Form objective.
    obj = Minimize(normInf(x))

    # Form and solve problem.
    prob = Problem(obj, constraints=constraint)
    prob.solve(solver=ECOS_BB)  # Returns the optimal value.
    # print "status:", prob.status
    # print "optimal value", prob.value
    # print "optimal var", theta.value
    # print "param", delta
    return theta.value
