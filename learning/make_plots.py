import convex as mycvx
import regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

seed = 4


def r2_excoeff_vs_time_cutoff(times):
    r2s = np.zeros_like(times, dtype='float64')
    num_data = np.zeros_like(times, dtype='int32')

    for index, time in enumerate(times):
        # get appropriate features
        x, y = regression.get_data(filenames=[
            '/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_excoeff_{}hr.csv'.format(
                time)])
        x = regression.Kau_MB_BR_features(x)

        # create the huber fit model
        alpha = 8
        model = mycvx.kfolds_convex(x, y, alpha, random_seed=seed)
        y_test = model['data']['y_test']
        y_pred = model['data']['y_pred']
        y_train = model['data']['y_train']
        y_train_pred = model['data']['y_train_pred']

        r2_test = np.round(r2_score(y_test, y_pred), 3)
        r2_train = np.round(r2_score(y_train, y_train_pred), 3)

        r2s[index] = r2_train
        num_data[index] = x.shape[0]

        print r2s, num_data
    return r2s, num_data


def r2_vs_time_cutoff(times, spm_cutoff=None):
    r2s = np.zeros_like(times, dtype='float64')
    num_data = np.zeros_like(times, dtype='int32')

    for index, time in enumerate(times):
        # get appropriate features
        x, y = regression.get_data(filenames=[
            '/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_{}hr.csv'.format(time)],
            spm_cutoff=spm_cutoff)
        x = regression.Kau_MB_BR_features(x)

        logy = np.log10(y)

        # create the huber fit model
        alpha = 8
        model = mycvx.kfolds_convex(x, logy, alpha, random_seed=seed)
        y_test = model['data']['y_test']
        y_pred = model['data']['y_pred']
        y_train = model['data']['y_train']
        y_train_pred = model['data']['y_train_pred']

        r2_test = np.round(r2_score(y_test, y_pred), 3)
        r2_train = np.round(r2_score(y_train, y_train_pred), 3)

        r2_test_unlog = np.round(r2_score(np.power(10, y_test), np.power(10, y_pred)), 3)
        r2_train_unlog = np.round(r2_score(np.power(10, y_train), np.power(10, y_train_pred)), 3)

        r2s[index] = r2_train_unlog
        num_data[index] = x.shape[0]

        print r2s, num_data
    return r2s, num_data


def plot_r2_over_time_diff(plot_excoeff=False):
    times = np.array([1, 2, 4, 8, 12, 16, 20, 24])
    if plot_excoeff:
        r2s, num_data = r2_excoeff_vs_time_cutoff(times)
    else:
        r2s, num_data = r2_vs_time_cutoff(times)
    # model on 5 band ratios and 6 reflectances
    # unlogged r2 below, ONLY ADJACENT CLOUD POINTS DELETED
    # r2s,num_data = [ 0.702  0.655  0.432  0.106  0.106  0.106  0.045  0.17 ], [ 61, 126, 234, 281, 281, 281, 298, 480]
    # unlogged r2 below, ALL POINTS UNDER CLOUD AND ADJACENT DELETED
    # [ 0.883  0.697  0.447  0.204  0.204  0.204  0.216  0.162] [ 34  75 157 185 185 185 197 302]

    # unlogged r2 below, NEW KAU FEATURES
    # [ 0.851  0.746  0.439  0.165  0.165  0.165  0.191  0.16 ] [ 34  75 157 185 185 185 197 302]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1plot = ax.plot(times, r2s, 'ob')

    ax2 = ax.twinx()
    ax2plot = ax2.plot(times, num_data, 'or')

    ax.set_xticks(times)

    ax.legend(ax1plot + ax2plot, ['R^2', 'Number of Samples'], loc='upper right')

    ax2.set_ylabel('Number of Samples')
    ydata = 'ExCoeff' if plot_excoeff else 'SPM'
    fig.suptitle(
        'R^2s of Robust Regression Model for {} versus Data Collection Time Difference Threshold'.format(ydata))
    ax.set_xlabel('Data Collection Time Difference Threshold (hrs)')
    ax.set_ylabel('R^2 of Robust Regression')

    ax.axis([0, 26, 0, 1])
    ax2.axis([0, 26, 0, 400])

    # print (max(np.max(y_pred), np.max(y_test))- np.min(np.min(y_pred), 0))*5./6. - np.min(np.min(y_pred), 0)
    # ax.text((max(np.max(y_pred), np.max(y_test))- min(np.min(y_pred), 0))*5./6. - min(np.min(y_pred), 0), np.max(y_test)/7.,  r'$R^2=%s$' % (r2train), fontsize=15)


    plt.savefig('../figures/huber_kau_training_{}_r2_vs_time'.format(ydata))
    # plt.show()


def make_huber_train_EA_MB(time_cutoff, spm_cutoff=None):
    # log10(SPM) = c0 + c1*x1 + c2*x2
    # x1 = Rrs(lambda1) + Rrs(lambda2) sensitive to spm
    # x2 = Rrs(lambda3)/Rrs(lambda1) compensating term
    # lambda 1 = green (555) = band2 for landsat 457, lambda 2 = red (670) = band3 for landsat 457,
    # lanbda3 = blue (490) = band1 for landsat 457
    x_names = ['reflec_1', 'reflec_2', 'reflec_3']
    y_names = ['Calculated SPM']

    filenames = ['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_2hr.csv']

    X, y = regression.get_data(x_names=x_names, y_names=y_names, filenames=filenames, Y_CODE='Calculated SPM')
    # X col 0 is band1, col 1 is band2, col 2 is band3
    # so lambda1 is col 1, lambda2 is col 2, lambda3 is col 0
    Rrs_lambda1 = X[:, 1]
    Rrs_lambda2 = X[:, 2]
    Rrs_lambda3 = X[:, 0]

    # log10(SPM) = c0 + c1*x1 + c2*x2
    # x1 = Rrs(lambda1) + Rrs(lambda2) sensitive to spm
    # x2 = Rrs(lambda3)/Rrs(lambda1) compensating term
    # lambda 1 = green (555) = band2 for landsat 457, lambda 2 = red (670) = band3 for landsat 457,
    # lanbda3 = blue (490) = band1 for landsat 457

    logy = np.log(y)

    X1 = Rrs_lambda1 + Rrs_lambda2
    X2 = np.divide(Rrs_lambda3, Rrs_lambda1)
    C0 = np.ones(Rrs_lambda1.size)

    x = np.array([C0, X1, X2]).T

    alpha = 8

    model = mycvx.kfolds_convex(x, logy, alpha, random_seed=seed)
    y_test = model['data']['y_test']
    y_pred = model['data']['y_pred']
    y_train = model['data']['y_train']
    y_train_pred = model['data']['y_train_pred']

    r2_test = np.round(r2_score(y_test, y_pred), 3)
    r2_train = np.round(r2_score(y_train, y_train_pred), 3)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(y_train_pred, y_train, '.b', label='Training Data')
    ax.plot(y_pred, y_test, '.r', label='Test Data')
    ax.plot(np.arange(0, 1.2 * max(np.max(y_test), np.max(y_train)), .1),
            np.arange(0, 1.2 * max(np.max(y_train), np.max(y_test)), .1), '-k')

    ax.legend()

    fig.suptitle(
        'Reconstruction Ability of Han Multi Band Regression Model using {}hr Data'.format(
            time_cutoff))
    ax.set_xlabel('Log Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('Log In situ measured SPM (mg/L)')

    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1. / 7.), s='Training Set R^2={}'.format(r2_train),
                textcoords="figure fraction", family='serif', horizontalalignment='right')
    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1.5 / 7.), s='Testing Set R^2={}'.format(r2_test),
                textcoords="figure fraction", family='serif', horizontalalignment='right')

    plt.savefig('../figures/huber_training_EA_MB_log_{}hr'.format(time_cutoff))
    # plt.show()
    print 'r2 train log: ', r2_train

    # unlog y_train and pred etc
    y_test = np.exp(y_test)
    y_pred = np.exp(y_pred)
    y_train = np.exp(y_train)
    y_train_pred = np.exp(y_train_pred)

    r2_test = np.round(r2_score(y_test, y_pred), 3)
    r2_train = np.round(r2_score(y_train, y_train_pred), 3)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_train_pred, y_train, '.b', label='Training Data')
    ax.plot(y_pred, y_test, '.r', label='Test Data')
    ax.legend()

    ax.plot(np.arange(0, 1.2 * max(np.max(y_test), np.max(y_train)), .1),
            np.arange(0, 1.2 * max(np.max(y_train), np.max(y_test)), .1), '-k')
    fig.suptitle(
        'Reconstruction Ability of Han Multi Band Model using {}hr Data'.format(
            time_cutoff))
    ax.set_xlabel('Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('In situ measure SPM (mg/L)')

    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1. / 7.), s='Training Set R^2={}'.format(r2_train),
                textcoords="figure fraction", family='serif', horizontalalignment='right')
    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1.5 / 7.), s='Testing Set R^2={}'.format(r2_test),
                textcoords="figure fraction", family='serif', horizontalalignment='right')

    plt.savefig('../figures/huber_training_EA_MB_{}hr'.format(time_cutoff))
    # plt.show()
    print 'r2 train not log: ', r2_train


def make_huber_train_basic(time_cutoff, spm_cutoff=None):
    '''
    :return:
    '''

    x, y = regression.get_data(filenames=[
        '/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_{}hr.csv'.format(time_cutoff)],
        spm_cutoff=spm_cutoff)  # 8hr data

    alpha = 8

    model = mycvx.kfolds_convex(x, y, alpha, random_seed=seed)
    y_test = model['data']['y_test']
    y_pred = model['data']['y_pred']
    y_train = model['data']['y_train']
    y_train_pred = model['data']['y_train_pred']

    r2_test = np.round(r2_score(y_test, y_pred), 3)
    r2_train = np.round(r2_score(y_train, y_train_pred), 3)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(y_train_pred, y_train, '.b', label='Training Data')
    ax.plot(y_pred, y_test, '.r', label='Test Data')
    ax.plot(np.arange(0, 1.2 * max(np.max(y_test), np.max(y_train)), .1),
            np.arange(0, 1.2 * max(np.max(y_train), np.max(y_test)), .1), '-k')

    ax.legend()

    fig.suptitle(
        'Reconstruction Ability of Multi Band Basic Regression Model on\n 6 Surface Reflectances using {}hr Data'.format(
            time_cutoff))
    ax.set_xlabel('Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('In situ measure SPM (mg/L)')

    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1. / 7.), s='Training Set R^2={}'.format(r2_train),
                textcoords="figure fraction", family='serif', horizontalalignment='right')
    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1.5 / 7.), s='Testing Set R^2={}'.format(r2_test),
                textcoords="figure fraction", family='serif', horizontalalignment='right')

    plt.savefig('../figures/huber_training_basic_{}hr'.format(time_cutoff))
    # plt.show()
    print 'r2 train log: ', r2_train


def make_huber_train_Kau_MB_ExCoeff(time_cutoff):
    '''
    log(ExCoeff) has low correlation to reflectances, plain excoeff has higher correlation
    Two hour correlation is 0.586 when using log(excoeff), 0.677 when using excoeff


    :param time_cutoff: acceptable maximum diffence in time between polaris and landsat
    :return: nothing, just saves plot
    '''
    # Get the appropriate features: 3 band ratios, 4 reflectances, SPM
    x, y = regression.get_data(filenames=[
        '/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_excoeff_{}hr.csv'.format(
            time_cutoff)], y_names=['Measured Extinction Coefficient'],
        Y_CODE='Measured Extinction Coefficient')  # 8hr data
    # print y
    x = regression.Kau_MB_BR_features(x)

    # generate the huber regression model
    alpha = 8
    model = mycvx.kfolds_convex(x, y, alpha, random_seed=seed)
    y_test = model['data']['y_test']
    y_pred = model['data']['y_pred']
    y_train = model['data']['y_train']
    y_train_pred = model['data']['y_train_pred']

    # Compute R2 scores
    r2_test = np.round(r2_score(y_test, y_pred), 3)
    r2_train = np.round(r2_score(y_train, y_train_pred), 3)

    # Start creating the plot for the log graph
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(y_train_pred, y_train, '.b', label='Training Data')
    ax.plot(y_pred, y_test, '.r', label='Test Data')
    ax.plot(np.arange(0, 1.2 * max(np.max(y_test), np.max(y_train)), .1),
            np.arange(0, 1.2 * max(np.max(y_train), np.max(y_test)), .1), '-k')

    ax.legend()

    fig.suptitle(
        'Extinction Coefficient Reconstruction Ability of Kau Multi Band Regression Model \n (3 Band Ratios & 4 Surface Reflectances) using {}hr Data'.format(
            time_cutoff))
    ax.set_xlabel('Remotely Sensed Extinction Coefficient')
    ax.set_ylabel('In situ measured Extinction Coefficient')

    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1. / 7.), s='Training Set R^2={}'.format(r2_train),
                textcoords="figure fraction", family='serif', horizontalalignment='right')
    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1.5 / 7.), s='Testing Set R^2={}'.format(r2_test),
                textcoords="figure fraction", family='serif', horizontalalignment='right')

    plt.savefig('../figures/huber_training_Kau_MB_excoeff_{}hr'.format(time_cutoff))


def make_huber_train_Kau_MB(time_cutoff, spm_cutoff=None):
    '''
    on 8hr data, the r2 for the log regression is .415 and .094 when unlogged
    on 4hr data, r2 is .313 on unlogged top 5 band ratios
    on 4hr data, r2 is .364 on unlogged when also using normal bands and top 3 ratios
    on 4hr data, r2 is .432 on unlogged when also using normal bands and top 5 ratios

    on 2hr data, r2 is .436 on unlogged top 5 band ratios
    on 2hr data, r2 is .621 on unlogged when also using normal bands and top 3 ratios
    on 2hr data, r2 is .655 on unlogged when also using normal bands and top 5 ratios

    :return:
    '''

    # Get the appropriate features: 3 band ratios, 4 reflectances, SPM
    x, y = regression.get_data(filenames=[
        '/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_{}hr.csv'.format(time_cutoff)],
        spm_cutoff=spm_cutoff)  # 8hr data
    x = regression.Kau_MB_BR_features(x)

    # log base 10 spm regression
    logy = np.log10(y)

    # generate the huber regression model
    alpha = 8
    model = mycvx.kfolds_convex(x, logy, alpha, random_seed=seed)
    y_test = model['data']['y_test']
    y_pred = model['data']['y_pred']
    y_train = model['data']['y_train']
    y_train_pred = model['data']['y_train_pred']
    # print model['theta']

    # Compute R2 scores
    r2_test = np.round(r2_score(y_test, y_pred), 3)
    r2_train = np.round(r2_score(y_train, y_train_pred), 3)

    # Start creating the plot for the log graph
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(y_train_pred, y_train, '.b', label='Training Data')
    ax.plot(y_pred, y_test, '.r', label='Test Data')
    ax.plot(np.arange(0, 1.2 * max(np.max(y_test), np.max(y_train)), .1),
            np.arange(0, 1.2 * max(np.max(y_train), np.max(y_test)), .1), '-k')

    ax.legend()

    fig.suptitle(
        'Reconstruction Ability of Kau Multi Band Regression Model \n (3 Band Ratios & 4 Surface Reflectances) using {}hr Data'.format(
            time_cutoff))
    ax.set_xlabel('Log Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('Log In situ measure SPM (mg/L)')

    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1. / 7.), s='Training Set R^2={}'.format(r2_train),
                textcoords="figure fraction", family='serif', horizontalalignment='right')
    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1.5 / 7.), s='Testing Set R^2={}'.format(r2_test),
                textcoords="figure fraction", family='serif', horizontalalignment='right')

    plt.savefig('../figures/huber_training_Kau_MB_log_{}hr'.format(time_cutoff))
    # plt.show()
    print 'r2 train log: ', r2_train

    # start creating the plot for the non log graph
    # unlog y_train, pred etc
    y_test = np.power(10, y_test)
    y_pred = np.power(10, y_pred)
    y_train = np.power(10, y_train)
    y_train_pred = np.power(10, y_train_pred)

    r2_test = np.round(r2_score(y_test, y_pred), 3)
    r2_train = np.round(r2_score(y_train, y_train_pred), 3)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_train_pred, y_train, '.b', label='Training Data')
    ax.plot(y_pred, y_test, '.r', label='Test Data')
    ax.legend()

    ax.plot(np.arange(0, 1.2 * max(np.max(y_test), np.max(y_train)), .1),
            np.arange(0, 1.2 * max(np.max(y_train), np.max(y_test)), .1), '-k')
    fig.suptitle(
        'Reconstruction Ability of Kau Multi Band Regression Model \n (3 Band Ratios & 4 Surface Reflectances) using {}hr Data'.format(
            time_cutoff))
    ax.set_xlabel('Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('In situ measure SPM (mg/L)')

    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1. / 7.), s='Training Set R^2={}'.format(r2_train),
                textcoords="figure fraction", family='serif', horizontalalignment='right')
    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1.5 / 7.), s='Testing Set R^2={}'.format(r2_test),
                textcoords="figure fraction", family='serif', horizontalalignment='right')

    plt.savefig('../figures/huber_training_Kau_MB_{}hr'.format(time_cutoff))
    # plt.show()
    print 'r2 train not log: ', r2_train


def make_huber_train_Kau_Simple(time_cutoff):
    # Get the appropriate features: 3 band ratios, 4 reflectances, SPM
    x, y = regression.get_data(filenames=[
        '/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_{}hr.csv'.format(
            time_cutoff)])  # 8hr data
    x = regression.Kau_Simple_features(x)

    # log base 10 spm regression
    logy = np.log10(y)

    # generate the huber regression model
    alpha = 8
    model = mycvx.kfolds_convex(x, logy, alpha, random_seed=seed)
    y_test = model['data']['y_test']
    y_pred = model['data']['y_pred']
    y_train = model['data']['y_train']
    y_train_pred = model['data']['y_train_pred']
    # print model['theta']

    # Compute R2 scores
    r2_test = np.round(r2_score(y_test, y_pred), 3)
    r2_train = np.round(r2_score(y_train, y_train_pred), 3)

    # Start creating the plot for the log graph
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(y_train_pred, y_train, '.b', label='Training Data')
    ax.plot(y_pred, y_test, '.r', label='Test Data')
    ax.plot(np.arange(0, 1.2 * max(np.max(y_test), np.max(y_train)), .1),
            np.arange(0, 1.2 * max(np.max(y_train), np.max(y_test)), .1), '-k')

    ax.legend()

    fig.suptitle(
        'Reconstruction Ability of Simple Kau Regression Model \n (3 Surface Reflectances) using {}hr Data'.format(
            time_cutoff))
    ax.set_xlabel('Log Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('Log In situ measure SPM (mg/L)')

    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1. / 7.), s='Training Set R^2={}'.format(r2_train),
                textcoords="figure fraction", family='serif', horizontalalignment='right')
    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1.5 / 7.), s='Testing Set R^2={}'.format(r2_test),
                textcoords="figure fraction", family='serif', horizontalalignment='right')

    plt.savefig('../figures/huber_training_Kau_Simple_log_{}hr'.format(time_cutoff))
    # plt.show()
    print 'r2 train log: ', r2_train

    # start creating the plot for the non log graph
    # unlog y_train, pred etc
    y_test = np.power(10, y_test)
    y_pred = np.power(10, y_pred)
    y_train = np.power(10, y_train)
    y_train_pred = np.power(10, y_train_pred)

    r2_test = np.round(r2_score(y_test, y_pred), 3)
    r2_train = np.round(r2_score(y_train, y_train_pred), 3)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_train_pred, y_train, '.b', label='Training Data')
    ax.plot(y_pred, y_test, '.r', label='Test Data')
    ax.legend()

    ax.plot(np.arange(0, 1.2 * max(np.max(y_test), np.max(y_train)), .1),
            np.arange(0, 1.2 * max(np.max(y_train), np.max(y_test)), .1), '-k')
    fig.suptitle(
        'Reconstruction Ability of Simple Kau Regression Model \n (3 Surface Reflectances) using {}hr Data'.format(
            time_cutoff))
    ax.set_xlabel('Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('In situ measure SPM (mg/L)')

    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1. / 7.), s='Training Set R^2={}'.format(r2_train),
                textcoords="figure fraction", family='serif', horizontalalignment='right')
    ax.annotate(xy=(0, 0), xytext=(5. / 6., 1.5 / 7.), s='Testing Set R^2={}'.format(r2_test),
                textcoords="figure fraction", family='serif', horizontalalignment='right')

    plt.savefig('../figures/huber_training_Kau_simple_{}hr'.format(time_cutoff))
    # plt.show()
    print 'r2 train not log: ', r2_train


def make_huber_test():
    x, y = regression.get_data()

    alpha = 8
    model = mycvx.kfolds_convex(x, y, alpha, random_seed=seed)
    y_test = model['data']['y_test']
    y_pred = model['data']['y_pred']
    r2 = np.round(r2_score(y_test, y_pred), 3)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_pred, y_test, 'or')
    ax.plot(np.arange(0, 1.05 * np.max(y_test), .1), np.arange(0, 1.05 * np.max(y_test), .1), '-k')
    fig.suptitle('Predictive Ability of Robust Regression Model')
    ax.set_xlabel('Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('In situ measure SPM (mg/L)')

    # print (max(np.max(y_pred), np.max(y_test))- np.min(np.min(y_pred), 0))*5./6. - np.min(np.min(y_pred), 0)
    ax.text((max(np.max(y_pred), np.max(y_test)) - min(np.min(y_pred), 0)) * 5. / 6. - min(np.min(y_pred), 0),
            np.max(y_test) / 7., r'$R^2=%s$' % (r2), fontsize=15)
    plt.savefig('../figures/huber_prediction')
    # plt.show()

    # model = mycvx.kfolds_convex(x, y, alpha, random_seed=seed, choice='ninf')
    # y_test = model['data']['y_test']
    # y_pred = model['data']['y_pred']
    # r2inf = np.round(r2_score(y_test, y_pred), 3)
    print 'huber r2: ', r2
    # print 'inf r2: ', r2inf


def make_huber_train():
    x, y = regression.get_data(
        filenames=['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_4hr.csv'])
    alpha = 8
    model = mycvx.kfolds_convex(x, y, alpha, random_seed=seed)
    y_test = model['data']['y_test']
    y_pred = model['data']['y_pred']
    y_train = model['data']['y_train']
    y_train_pred = model['data']['y_train_pred']

    r2 = np.round(r2_score(y_test, y_pred), 3)
    r2train = np.round(r2_score(y_train, y_train_pred), 3)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_train_pred, y_train, '.b')
    ax.plot(y_pred, y_test, '.r')
    ax.plot(np.arange(0, 1.2 * np.max(y_test), .1), np.arange(0, 1.2 * np.max(y_test), .1), '-k')
    fig.suptitle('Reconstruction Ability of Robust Regression Model')
    ax.set_xlabel('Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('In situ measure SPM (mg/L)')

    # print (max(np.max(y_pred), np.max(y_test))- np.min(np.min(y_pred), 0))*5./6. - np.min(np.min(y_pred), 0)
    ax.text((max(np.max(y_pred), np.max(y_test)) - min(np.min(y_pred), 0)) * 5. / 6. - min(np.min(y_pred), 0),
            np.max(y_test) / 7., r'$R^2=%s$' % (r2train), fontsize=15)
    plt.savefig('../figures/huber_training')
    # plt.show()
    print 'r2: ', r2train


if __name__ == '__main__':
    # make_huber_train()
    # make_huber_test()
    # [make_huber_train_Kau_MB(i) for i in [1,2,4,8]]
    # make_huber_train_Kau_MB(2)
    # make_huber_train_EA_MB(2)
    # make_huber_train_basic(2)
    # plot_r2_over_time_diff(plot_excoeff=False)
    # r2_vs_time_cutoff([8])
    make_huber_train_Kau_MB_ExCoeff(2)
    # make_huber_train_Kau_Simple(1)
