import numpy as np, pickle
from catObj import catObj
import pylab as pyl
import glob,pickle
from sklearn import cluster,mixture,decomposition,neighbors
from astropy.visualization import interval
from stsci import numdisplay
from astropy.io import fits
import matplotlib.gridspec as gridspec

def showSources(pred,X,Y,fits_fn):
    with fits.open(fits_fn) as han:
        data = han[0].data
        if data is None:
            data = han[1].data

    w = np.where(pred == 1)

    nsp = int(len(w[0])**0.5)
    if len(w[0])>nsp*nsp:
        nsp += 1

    fig = pyl.figure('good'+fits_fn,figsize=(15,15))
    fig.subplots_adjust(hspace=0,wspace=0)
    gs = gridspec.GridSpec(nsp,nsp)
    for ii in range(nsp):
        for jj in range(nsp):
            if ii*nsp+jj<len(w[0]):
                x,y = int(X[w[0]][ii*nsp+jj]-1),int(Y[w[0]][ii*nsp+jj]-1)
                cut = data[y-10:y+11,x-10:x+11]
                (z1,z2) = numdisplay.zscale.zscale(cut,contrast = 0.5)
                normer = interval.ManualInterval(z1,z2)
                Cut = normer(cut)
                sp = pyl.subplot(gs[ii,jj])
                pyl.imshow(Cut)

    w = np.where(pred == -1)
    nsp = int(len(w[0])**0.5)
    if len(w[0])>nsp*nsp:
        nsp += 1

    fig = pyl.figure('bad'+fits_fn,figsize=(15,15))
    fig.subplots_adjust(hspace=0,wspace=0)
    gs = gridspec.GridSpec(nsp,nsp)
    for ii in range(nsp):
        for jj in range(nsp):
            if ii*nsp+jj<len(w[0]):
                x,y = int(X[w[0]][ii*nsp+jj]-1),int(Y[w[0]][ii*nsp+jj]-1)
                cut = data[y-10:y+11,x-10:x+11]
                (z1,z2) = numdisplay.zscale.zscale(cut,contrast = 0.5)
                normer = interval.ManualInterval(z1,z2)
                Cut = normer(cut)
                sp = pyl.subplot(gs[ii,jj])
                pyl.imshow(Cut)
    #pyl.show()


def skneigh(sample,fit = None, contamination = 0.05):
    if fit is None:
        lof = neighbors.LocalOutlierFactor(contamination=contamination)
        return lof.fit_predict(sample)
    else:
        """
        Assume the sample contains only good points now
        """
        lof = neighbors.LocalOutlierFactor(contamination=0.00001,novelty = True)
        lof.fit(sample)
        return lof.predict(fit)

def skpca(sample):
    PCA = decomposition.PCA(n_components = 2, copy = True)
    PCA.fit(sample)
    t = PCA.transform(sample)
    return t


def skcluster(sample):
    gmm = mixture.GaussianMixture(
        n_components=2, covariance_type='full')
    gmm.fit(sample)
    pred = gmm.predict(sample)
    spectral = cluster.SpectralClustering(
        n_clusters=3, eigen_solver='arpack',
        affinity="nearest_neighbors")
    spectral.fit(sample)
    pred = spectral.labels_.astype(np.int)
    return pred



if __name__ == "__main__":
    fits_fn = 'CORR-0132546-034.fits'
    fn =fits_fn.replace('.fits','.psfLikelihoods')
    contamination = 0.01

    with open(fn) as han:
        [realLikelihoods,realVs,realLikelihoods_small,realVs_small,realX,realY,likelihoods,likelihoods_small,X,Y,V] = pickle.load(han)

    sample = np.zeros((len(likelihoods),2)).astype('float64')
    sample[:,0] = likelihoods
    sample[:,1] = likelihoods_small

    sample[:,1] -= sample[:,0]

    pred = skneigh(sample, contamination = contamination)

    w = np.where((sample[:,1]>-330))#&(sample[:,0]>-24))
    W = np.where((sample[:,1]<-330))#|(sample[:,0]<-24))
    pred[w] = 1
    pred[W] = -1

    """
    w = np.where(pred == -1)
    print V[w]
    w = np.where(pred == 1)
    print V[w]
    exit()
    """
    culled_sample = []
    for i in range(len(pred)):
        if pred[i] == 1:
            culled_sample.append(sample[i])
    culled_sample = np.array(culled_sample)

    showCulledPlanted = True
    if showCulledPlanted:

        showSources(pred,X,Y,fits_fn.replace('.fits','_planted.fits'))

    real_sample = np.zeros((len(realLikelihoods),2)).astype('float64')
    real_sample[:,0] = realLikelihoods
    real_sample[:,1] = realLikelihoods_small

    real_pred =  skneigh(culled_sample,real_sample, contamination = contamination)


    showCulledReal = True
    if showCulledReal:

        showSources(real_pred,realX,realY,fits_fn)
    """
    t = skpca(sample)


    median = np.median(t,axis=0)
    std = (np.sqrt(np.mean((t[:,0] - median[0])**2)),np.sqrt(np.mean((t[:,1] - median[1])**2)))
    w = np.where( (np.abs(t[:,0]-median[0])>3*std[0]) | (np.abs(t[:,1]-median[1])>3*std[1]) )
    print median,std
    print w
    pred = np.zeros(len(t))
    pred[w] = 1

    pred = skcluster(t)
    print pred
    """



    fig = pyl.figure(10)
    sp1 = fig.add_subplot(211)
    sp2 = fig.add_subplot(212)
    c = ['r','b','g','y']
    for i,p in enumerate(np.unique(pred)):
        w = np.where(pred == p)
        sp1.scatter(sample[:,0][w],sample[:,1][w]-sample[:,0][w],c=c[i])
        sp1.set_title('Planted Sources')
    for i,p in enumerate(np.unique(real_pred)):
        w = np.where(real_pred == p)
        sp2.scatter(real_sample[:,0][w],real_sample[:,1][w]-real_sample[:,0][w],c=c[i])
        sp2.set_title('Real Sources')
        #sp2.scatter(t[:,0][w],t[:,1][w],c=c[i])
    pyl.show()
