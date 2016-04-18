import convex as mycvx
import regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
seed = 4

def r2_vs_time_cutoff(times):
    # times = np.array([1,2,4])

    r2s = np.zeros_like(times,dtype='float64')
    num_data = np.zeros_like(times,dtype='int32')

    for index,time in enumerate(times):
        # print index,time
        x,y=regression.get_data(filenames=['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_{}hr.csv'.format(time)])
        x = regression.division_feature_expansion(x)
        # ONLY top 5 bands
        x = x[:,[29,9,14,28,5]]
        logy = np.log(y)
        alpha = 8

        model = mycvx.kfolds_convex(x, logy, alpha, random_seed=seed)
        y_test = model['data']['y_test']
        y_pred = model['data']['y_pred']
        y_train = model['data']['y_train']
        y_train_pred = model['data']['y_train_pred']


        r2_test = np.round(r2_score(y_test, y_pred), 3)
        r2_train = np.round(r2_score(y_train, y_train_pred), 3)
        # print r2_train,r2_test
        r2s[index] = r2_train
        num_data[index] = x.shape[0]

        print r2s,num_data
    return r2s,num_data

def plot_r2_over_time_diff():
    times = np.array([1,2,4,8,12,16,20,24])
    # r2s,num_data = r2_vs_time_cutoff(times)
    r2s,num_data = [ 0.508,  0.54,   0.516,  0.415,  0.415,  0.415,  0.433,  0.417], [ 61, 126, 234, 281, 281, 281, 298, 480]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1plot = ax.plot(times, r2s, 'ob')

    ax2 = ax.twinx()
    ax2plot = ax2.plot(times,num_data,'or')

    ax.set_xticks([1,2,4,6,8,12,16,20,24])

    ax.legend(ax1plot+ax2plot,['R^2','Number of Samples'],loc=2)
    # ax2.legend('Num Samples')
    # ax.legend(loc=1)
    # ax2.legend(loc=2)

    ax2.set_ylabel('Number of Samples')
    fig.suptitle('R^2s of Robust Regression Model versus Data Collection Time Difference Threshold')
    ax.set_xlabel('Data Collection Time Difference Threshold (hrs)')
    ax.set_ylabel('R^2 of Robust Regression')

    ax.axis([0,26,0,.8])
    ax2.axis([0,26,0,500])

    # print (max(np.max(y_pred), np.max(y_test))- np.min(np.min(y_pred), 0))*5./6. - np.min(np.min(y_pred), 0)
    #ax.text((max(np.max(y_pred), np.max(y_test))- min(np.min(y_pred), 0))*5./6. - min(np.min(y_pred), 0), np.max(y_test)/7.,  r'$R^2=%s$' % (r2train), fontsize=15)
    plt.savefig('../figures/huber_training_r2_vs_time')
    # plt.show()


def make_huber_train_band_ratios():
    '''
    on 8hr data, the r2 for the log regression is .555 and .21 when unlogged
    :return:
    '''

    x,y=regression.get_data(filenames=['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_2hr.csv']) # 8hr data
    x = regression.division_feature_expansion(x)

    # ONLY top 5 bands
    x = x[:,[29,9,14,28,5]]

    # log spm regression
    logy = np.log(y)

    alpha = 8
    model = mycvx.kfolds_convex(x, logy, alpha, random_seed=seed)
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
    ax.plot(np.arange(0, 1.2*np.max(y_test), .1), np.arange(0, 1.2*np.max(y_test), .1), '-k')
    fig.suptitle('Reconstruction Ability of Robust Regression Model on 5 Most Correlated Band Ratios')
    ax.set_xlabel('Log Remotely Sensed SPM (g/ml)')
    ax.set_ylabel('Log In situ measure SPM (g/ml)')

    # print (max(np.max(y_pred), np.max(y_test))- np.min(np.min(y_pred), 0))*5./6. - np.min(np.min(y_pred), 0)
    ax.text((max(np.max(y_pred), np.max(y_test))- min(np.min(y_pred), 0))*5./6. - min(np.min(y_pred), 0), np.max(y_test)/7.,  r'$R^2=%s$' % (r2train), fontsize=15)
    plt.savefig('../figures/huber_training_band_ratio_log')
    # plt.show()
    print 'r2: ', r2train

    # unlog y_train and pred etc
    y_test = np.exp(y_test)
    y_pred = np.exp(y_pred)
    y_train = np.exp(y_train)
    y_train_pred = np.exp(y_train_pred)

    r2 = np.round(r2_score(y_test, y_pred), 3)
    r2train = np.round(r2_score(y_train, y_train_pred), 3)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_train_pred, y_train, '.b')
    ax.plot(y_pred, y_test, '.r')
    ax.plot(np.arange(0, 1.2*np.max(y_test), .1), np.arange(0, 1.2*np.max(y_test), .1), '-k')
    fig.suptitle('Reconstruction Ability of Robust Regression Model on 5 Most Correlated Band Ratios')
    ax.set_xlabel('Remotely Sensed SPM (g/ml)')
    ax.set_ylabel('In situ measure SPM (g/ml)')

    # print (max(np.max(y_pred), np.max(y_test))- np.min(np.min(y_pred), 0))*5./6. - np.min(np.min(y_pred), 0)
    ax.text((max(np.max(y_pred), np.max(y_test))- min(np.min(y_pred), 0))*5./6. - min(np.min(y_pred), 0), np.max(y_test)/7.,  r'$R^2=%s$' % (r2train), fontsize=15)
    plt.savefig('../figures/huber_training_band_ratio')
    # plt.show()
    print 'r2: ', r2train


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
    ax.plot(np.arange(0, 1.05*np.max(y_test), .1), np.arange(0, 1.05*np.max(y_test), .1), '-k')
    fig.suptitle('Predictive Ability of Robust Regression Model')
    ax.set_xlabel('Remotely Sensed SPM (g/ml)')
    ax.set_ylabel('In situ measure SPM (g/ml)')

    # print (max(np.max(y_pred), np.max(y_test))- np.min(np.min(y_pred), 0))*5./6. - np.min(np.min(y_pred), 0)
    ax.text((max(np.max(y_pred), np.max(y_test))- min(np.min(y_pred), 0))*5./6. - min(np.min(y_pred), 0), np.max(y_test)/7.,  r'$R^2=%s$' % (r2), fontsize=15)
    plt.savefig('../figures/huber_prediction')
    # plt.show()

    # model = mycvx.kfolds_convex(x, y, alpha, random_seed=seed, choice='ninf')
    # y_test = model['data']['y_test']
    # y_pred = model['data']['y_pred']
    # r2inf = np.round(r2_score(y_test, y_pred), 3)
    print 'huber r2: ', r2
    # print 'inf r2: ', r2inf

def make_huber_train():
    x, y = regression.get_data()
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
    ax.plot(y_train_pred, y_train, 'ob')
    ax.plot(y_pred, y_test, 'or')
    ax.plot(np.arange(0, 1.2*np.max(y_test), .1), np.arange(0, 1.2*np.max(y_test), .1), '-k')
    fig.suptitle('Reconstruction Ability of Robust Regression Model')
    ax.set_xlabel('Remotely Sensed SPM (g/ml)')
    ax.set_ylabel('In situ measure SPM (g/ml)')

    # print (max(np.max(y_pred), np.max(y_test))- np.min(np.min(y_pred), 0))*5./6. - np.min(np.min(y_pred), 0)
    ax.text((max(np.max(y_pred), np.max(y_test))- min(np.min(y_pred), 0))*5./6. - min(np.min(y_pred), 0), np.max(y_test)/7.,  r'$R^2=%s$' % (r2train), fontsize=15)
    plt.savefig('../figures/huber_training')
    # plt.show()
    print 'r2: ', r2train

if __name__ == '__main__':
    # make_huber_train()
    # make_huber_test()
    # make_huber_train_band_ratios()
    plot_r2_over_time_diff()
