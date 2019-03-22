
from __future__ import print_function


# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

import numpy as np
import matplotlib.pyplot as pyl

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from time import time
from astropy.io import fits

# #############################################################################
# Setting up

fn = 'CORR-0132546-072.fits'

# Load Data
with fits.open('test_image_data/'+fn.replace('.fits','_trainingset.fits')) as han:
    flat = han[0].data
    snr = han[1].data
    header = han[0].header


#flat = np.clip(flat,header['BGMEAN'],np.max(flat))-header['BGMEAN']
X = flat/np.max(flat)
(a,b) = X.shape


snr_ranges = [5,15,30,70] # this at first glance seems to work well
snr_ranges = [5,25,50,100000]

Y = np.zeros((len(snr))).astype('int')
for i in range(len(snr_ranges)-1):

    w = np.where((snr>snr_ranges[i])&(snr<snr_ranges[i+1]))
    Y[w] = i+1
    w = np.where((-snr>snr_ranges[i])&(-snr<snr_ranges[i+1]))
    Y[w] = -(i+1)



#now normalize each to its max brightness
for y in np.unique(Y):
    w = np.where(Y==y)
    print(len(w[0]))
    m = np.max(X[w])
    print(y,m)
    X[w] = X[w]/m
Y[np.where(Y<=0)] = 0

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.35, random_state=0)

(n_samples,h2) = X.shape
h = int(h2**0.5)
w = h


#PCA models
n_components = 50

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))
#for i in range(n_components):
#    pyl.imshow(eigenfaces[i])
#    pyl.show()

print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
clf = clf.fit(X_train_pca, Y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
Y_pred_svm = clf.predict(X_test_pca)

print(metrics.classification_report(np.copy(Y_test), np.copy(Y_pred_svm)))



for i in range(len(np.unique(Y))):
    print(i,len(np.where(Y_test==i)[0]))

print(metrics.confusion_matrix(np.copy(Y_test), np.copy(Y_pred_svm), labels=range(len(snr_ranges))))




#logistic regression models



# Models we will use
logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000,
                                           multi_class='multinomial')
rbm = BernoulliRBM(random_state=0, verbose=True)


rbm_features_classifier = Pipeline(
    steps=[('rbm', rbm), ('logistic', logistic)])

# #############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000

# Training RBM-Logistic Pipeline
#rbm_features_classifier.fit(X_train, Y_train)

# Training the Logistic regression classifier directly on the pixel
raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 10.
raw_pixel_classifier.fit(X_train, Y_train)


# #############################################################################
# Evaluation

#Y_pred = rbm_features_classifier.predict(X_test)
#print("Logistic regression using RBM features:\n%s\n" % (
#    metrics.classification_report(Y_test, Y_pred)))

Y_pred_rpc = raw_pixel_classifier.predict(X_test)

print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(np.copy(Y_test), np.copy(Y_pred_rpc))))
print(metrics.confusion_matrix(np.copy(Y_test), np.copy(Y_pred_rpc), labels=range(len(snr_ranges))))


#for i in range(len(Y_test)):
#    if Y_test[i] >0 and Y_pred_rpc[i]==0 and Y_pred_svm[i]==0:
#        print(Y_test[i],Y_pred_rpc[i],Y_pred_svm[i])
#        pyl.imshow(X_test[i].reshape(17,17))
#        pyl.show()

for i in range(len(Y_test)):
    if Y_pred_rpc[i]==0 and Y_pred_svm[i]==0:
        print(Y_test[i],Y_pred_rpc[i],Y_pred_svm[i])
        pyl.imshow(X_test[i].reshape(17,17))
        pyl.show()
exit()
# #############################################################################
# Plotting

fig1 = plt.figure('Example',figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

fig2 = plt.figure('Unsupervised',figsize=(4.2, 4))
for i, comp in enumerate(RBM.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by unsupervised RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
