import convex as mycvx
import regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
seed = 4

def make_huber_train_band_ratios():
    x,y=regression.get_data() # 8hr data
    x = regression.division_feature_expansion(x)

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
    ax.plot(y_train_pred, y_train, 'ob')
    ax.plot(y_pred, y_test, 'or')
    ax.plot(np.arange(0, 1.2*np.max(y_test), .1), np.arange(0, 1.2*np.max(y_test), .1), '-k')
    fig.suptitle('Reconstruction Ability of Robust Regression Model')
    ax.set_xlabel('Remotely Sensed SPM (g/ml)')
    ax.set_ylabel('In situ measure SPM (g/ml)')

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
    ax.plot(y_train_pred, y_train, 'ob')
    ax.plot(y_pred, y_test, 'or')
    ax.plot(np.arange(0, 1.2*np.max(y_test), .1), np.arange(0, 1.2*np.max(y_test), .1), '-k')
    fig.suptitle('Reconstruction Ability of Robust Regression Model')
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
    make_huber_train()
    make_huber_test()
    make_huber_train_band_ratios()

