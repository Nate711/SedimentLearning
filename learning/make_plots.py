import convex as mycvx
import regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
seed = 4

def r2_vs_time_cutoff(times,spm_cutoff=-1):
    # times = np.array([1,2,4])

    r2s = np.zeros_like(times,dtype='float64')
    num_data = np.zeros_like(times,dtype='int32')

    thetas = np.array([]).reshape(9,0)
    for index,time in enumerate(times):
        # print index,time
        x,y=regression.get_data(filenames=['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_{}hr.csv'.format(time)],spm_cutoff=spm_cutoff)

        top_5_bands = regression.top_5_band_ratios(x)
        # Add to feature array
        x = np.append(x,top_5_bands,axis=1)

        logy = np.log(y)
        alpha = 8

        model = mycvx.kfolds_convex(x, logy, alpha, random_seed=seed)
        y_test = model['data']['y_test']
        y_pred = model['data']['y_pred']
        y_train = model['data']['y_train']
        y_train_pred = model['data']['y_train_pred']


        r2_test = np.round(r2_score(y_test, y_pred), 3)
        r2_train = np.round(r2_score(y_train, y_train_pred), 3)

        r2_test_unlog = np.round(r2_score(np.exp(y_test),np.exp(y_pred)),3)
        r2_train_unlog = np.round(r2_score(np.exp(y_train),np.exp(y_train_pred)),3)

        # print r2_train,r2_test
        # r2s[index] = r2_train
        r2s[index] = r2_train_unlog
        num_data[index] = x.shape[0]

        # TODO fix this bug, thetas is 1d array
        thetas = np.append(thetas,model['theta'],axis=1)

        print r2s,num_data
    print thetas
    # TODO, fix division
    print np.divide(thetas[1],thetas[0])
    return r2s,num_data

def plot_r2_over_time_diff():
    times = np.array([1,2,4,8,12,16,20,24])
    r2s,num_data = r2_vs_time_cutoff(times)
    # model on 5 band ratios and 6 reflectances
    # unlogged r2 below, ONLY ADJACENT CLOUD POINTS DELETED
    # r2s,num_data = [ 0.702  0.655  0.432  0.106  0.106  0.106  0.045  0.17 ], [ 61, 126, 234, 281, 281, 281, 298, 480]
    # unlogged r2 below, ALL POINTS UNDER CLOUD AND ADJACENT DELETED
    # [ 0.883  0.697  0.447  0.204  0.204  0.204  0.216  0.162] [ 34  75 157 185 185 185 197 302]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1plot = ax.plot(times, r2s, 'ob')

    ax2 = ax.twinx()
    ax2plot = ax2.plot(times,num_data,'or')

    ax.set_xticks(times)

    ax.legend(ax1plot+ax2plot,['R^2','Number of Samples'],loc='upper right')
    # ax2.legend('Num Samples')
    # ax.legend(loc=1)
    # ax2.legend(loc=2)

    ax2.set_ylabel('Number of Samples')
    fig.suptitle('R^2s of Robust Regression Model versus Data Collection Time Difference Threshold')
    ax.set_xlabel('Data Collection Time Difference Threshold (hrs)')
    ax.set_ylabel('R^2 of Robust Regression')

    ax.axis([0,26,0,.9])
    ax2.axis([0,26,0,500])

    # print (max(np.max(y_pred), np.max(y_test))- np.min(np.min(y_pred), 0))*5./6. - np.min(np.min(y_pred), 0)
    #ax.text((max(np.max(y_pred), np.max(y_test))- min(np.min(y_pred), 0))*5./6. - min(np.min(y_pred), 0), np.max(y_test)/7.,  r'$R^2=%s$' % (r2train), fontsize=15)
    plt.savefig('../figures/huber_training_r2_vs_time')
    # plt.show()

def make_huber_train_band_ratios(time_cutoff,spm_cutoff=-1):
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

    x,y=regression.get_data(filenames=['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_{}hr.csv'.format(time_cutoff)],spm_cutoff=spm_cutoff) # 8hr data

    # Get top 5 correlated band ratios
    # top_5_bands = regression.division_feature_expansion(x)[:,[29,9,14,28,5]]
    top_5_bands = regression.top_5_band_ratios(x)
    # Add to feature array
    x = np.append(x,top_5_bands,axis=1)

    '''
    # Get top 3 correlated band ratios
    top_3_bands = regression.division_feature_expansion(x)[:,[29,9,14,28,5]]
    # Add to feature array
    x = np.append(x,top_3_bands,axis=1)
    '''

    # log spm regression
    logy = np.log(y)

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

    ax.plot(y_train_pred, y_train, '.b',label='Training Data')
    ax.plot(y_pred, y_test, '.r',label='Test Data')
    ax.plot(np.arange(0, 1.2*max(np.max(y_test),np.max(y_train)), .1), np.arange(0, 1.2*max(np.max(y_train),np.max(y_test)), .1), '-k')

    ax.legend()

    fig.suptitle('Reconstruction Ability of Robust Regression Model on\n 5 Most Correlated Band Ratios and 6 Surface Reflectances using {}hr Data'.format(time_cutoff))
    ax.set_xlabel('Log Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('Log In situ measure SPM (mg/L)')

    ax.annotate(xy=(0,0),xytext=(5./6.,1./7.),s='Training Set R^2={}'.format(r2_train),textcoords="figure fraction",family='serif',horizontalalignment='right')
    ax.annotate(xy=(0,0),xytext=(5./6.,1.5/7.),s='Testing Set R^2={}'.format(r2_test),textcoords="figure fraction",family='serif',horizontalalignment='right')

    plt.savefig('../figures/huber_training_band_ratio_log_{}hr'.format(time_cutoff))
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
    ax.plot(y_train_pred, y_train, '.b',label='Training Data')
    ax.plot(y_pred, y_test, '.r',label='Test Data')
    ax.legend()

    ax.plot(np.arange(0, 1.2*max(np.max(y_test),np.max(y_train)), .1), np.arange(0, 1.2*max(np.max(y_train),np.max(y_test)), .1), '-k')
    fig.suptitle('Reconstruction Ability of Robust Regression Model on \n5 Most Correlated Band Ratios and 6 Surface Reflectances using {}hr Data'.format(time_cutoff))
    ax.set_xlabel('Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('In situ measure SPM (mg/L)')

    ax.annotate(xy=(0,0),xytext=(5./6.,1./7.),s='Training Set R^2={}'.format(r2_train),textcoords="figure fraction",family='serif',horizontalalignment='right')
    ax.annotate(xy=(0,0),xytext=(5./6.,1.5/7.),s='Testing Set R^2={}'.format(r2_test),textcoords="figure fraction",family='serif',horizontalalignment='right')

    plt.savefig('../figures/huber_training_band_ratio_{}hr'.format(time_cutoff))
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
    ax.plot(np.arange(0, 1.05*np.max(y_test), .1), np.arange(0, 1.05*np.max(y_test), .1), '-k')
    fig.suptitle('Predictive Ability of Robust Regression Model')
    ax.set_xlabel('Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('In situ measure SPM (mg/L)')

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
    x, y = regression.get_data(filenames=['/Users/Nathan/Dropbox/SedimentLearning/data/landsat_polaris_filtered/filtered_4hr.csv'])
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
    ax.plot(np.arange(0, 1.2*np.max(y_test), .1), np.arange(0, 1.2*np.max(y_test), .1), '-k')
    fig.suptitle('Reconstruction Ability of Robust Regression Model')
    ax.set_xlabel('Remotely Sensed SPM (mg/L)')
    ax.set_ylabel('In situ measure SPM (mg/L)')

    # print (max(np.max(y_pred), np.max(y_test))- np.min(np.min(y_pred), 0))*5./6. - np.min(np.min(y_pred), 0)
    ax.text((max(np.max(y_pred), np.max(y_test))- min(np.min(y_pred), 0))*5./6. - min(np.min(y_pred), 0), np.max(y_test)/7.,  r'$R^2=%s$' % (r2train), fontsize=15)
    plt.savefig('../figures/huber_training')
    # plt.show()
    print 'r2: ', r2train

if __name__ == '__main__':
    # make_huber_train()
    # make_huber_test()
    # [make_huber_train_band_ratios(i) for i in [1,2,4,8]]
    # make_huber_train_band_ratios(2)
    plot_r2_over_time_diff()
    # r2_vs_time_cutoff([8])