from sklearn import linear_model, cross_validation, decomposition
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import csv

def get_data(filename,x_names,y_name):
    x_dict = {}
    y_values = np.array([],dtype='float64')

    with open(filename,'rb') as csvfile:
        reader = csv.DictReader(csvfile,delimiter=',')

        for n in x_names:
            x_dict[n] = np.array([],dtype='float64')

        for row in reader:
            for key in row.keys():
                if key in x_names:
                    x_dict[key] = np.append(x_dict[key],float(row[key]))

                if key == y_name:
                    y_values = np.append(y_values,float(row[key]))

    x_array = np.array(x_dict.values())

    return x_array.T,y_values

X,y = get_data('mlr03.csv',['EXAM1','EXAM2','EXAM3'],'FINAL')

print X.shape,y.shape

log_alphas = np.array(np.arange(-20,20,0.5),dtype='float64')

alphas = 2**log_alphas

clf = linear_model.RidgeCV(alphas = alphas,cv=None,store_cv_values=True) # MSE = 7.76
#clf = linear_model.LassoCV(alphas = alphas) # MSE = 5.8

clf = svm.LinearSVR() # RME around 14

clf.fit(X,y)


#plt.plot(clf.cv_values_)
#plt.show()

y_predict = clf.predict(X)

plt.plot(y_predict, y,'ok')
#plt.title(r'Ridge Regression $\alpha$ = %f' % (clf.alpha_))
plt.xlabel('Predicted Final Score')
plt.ylabel('Actual Final Score')
plt.show()

#print clf.alpha_
print clf.coef_
print mean_squared_error(y,y_predict)






