import trippy, numpy as np, pylab as pyl, scipy as sci
from trippy import psf, psfStarChooser, MCMCfit, scamp
from astropy.io import fits
from stsci import numdisplay
from astropy.visualization import interval
import time
from scipy import optimize as opti
import pickle
from paths import *

def resid(p,cutout,boxWidth = 7,verbose=False):
    (x,y,m) = p
    xt,yt = int(x),int(y)
    (a,b)=cutout.shape
    res = goodPSF.remove(x,y,m,cutout,useLinePSF=False)[yt-boxWidth:yt+boxWidth+1,xt-boxWidth:xt+boxWidth+1]
    if verbose:
        print np.sum(res**2)**0.5,x,y,m

    return np.array(res).astype('float').reshape((boxWidth*2+1)**2)

def likelihood(p,cutout,bg,boxWidth = 7,verbose=False):
    (x,y,m) = p
    res = resid(p,cutout-bg,boxWidth = boxWidth,verbose=verbose)
    xt,yt = int(x),int(y)
    ue2 = np.abs(cutout[yt-boxWidth:yt+boxWidth+1,xt-boxWidth:xt+boxWidth+1].reshape((boxWidth*2+1)**2))
    return -0.5*np.sum(res**2/ue2)


apNum = 4

#good seeing image
fn = '/media/fraserw/rocketdata/SEP2017/02093/sexSaves/CORR-0132546-034.cat'
fits_fn = 'CORR-0132546-034.fits'
with fits.open(fits_fn) as han:
    data = han[1].data
    header = han[0].header
data += header['BGMEAN']
gain = header['MEDGAIN']

catalog = scamp.getCatalog(fn,paramFile='sextract.param')
w = np.where((catalog['X_IMAGE']>50) & (catalog['X_IMAGE']<1995) & (catalog['Y_IMAGE']>50) & (catalog['Y_IMAGE']<4123) \
        &  ((catalog['FLUX_APER(5)'][:,apNum]/catalog['FLUXERR_APER(5)'][:,apNum])>5) \
        &  ((catalog['FLUX_APER(5)'][:,apNum]/catalog['FLUXERR_APER(5)'][:,apNum])<50) & (catalog['FWHM_IMAGE']>1.5))
lowSources = np.zeros((len(w[0]),4)).astype('float64')
lowSources[:,0] = catalog['X_IMAGE'][w]
lowSources[:,1] = catalog['Y_IMAGE'][w]
lowSources[:,2] = catalog['FLUXERR_APER(5)'][:,apNum][w]
lowSources[:,3] = catalog['FLUX_APER(5)'][:,apNum][w]

with open(fits_fn.replace('.fits','.psfStars')) as han:
    dat = han.readlines()
sources = []
for i in range(len(dat)):
    s = dat[i].split()
    sources.append([float(s[0]),float(s[1]),float(s[2]),float(s[3])])
sources = np.array(sources)


genPSF = False
if genPSF:
    psfWidth = int(6*4.5)
    starChooser = psfStarChooser.starChooser(data,
                                             sources[:,0],sources[:,1],
                                             sources[:,3],sources[:,2])

    (goodFits,goodMeds,goodSTDs) = starChooser(30,1,noVisualSelection=True,autoTrim=False)
    goodPSF=psf.modelPSF(np.arange(psfWidth*2+1),np.arange(psfWidth*2+1), alpha=goodMeds[2],beta=goodMeds[3],repFact=10)
    goodPSF.genLookupTable(data,goodFits[:,4],goodFits[:,5],verbose=False)
    fwhm=goodPSF.FWHM()
    print fwhm
    #(z1,z2)=numdisplay.zscale.zscale(goodPSF.lookupTable)
    #normer=interval.ManualInterval(z1,z2)
    #pyl.imshow(normer(goodPSF.lookupTable))
    #pyl.show()

    goodPSF.psfStore(fits_fn.replace('.fits','.psf.fits'))
else:
    goodPSF = psf.modelPSF(restore=fits_fn.replace('.fits','.psf.fits'))


#measuring faint sources for training set generation
fwhm = goodPSF.FWHM()
fitWidths = [int(fwhm)+1,int(2*fwhm)]
ftol = 1.e-3 #1.e-8

#planting point sources for training set generation
nplant = 300
(A,B) = data.shape
X = sci.rand(nplant)*(B-60.0)+30.0#np.arange(50.,100.,50.)
Y = sci.rand(nplant)*(A-60.0)+30.0
V = sci.rand(nplant)*1.0+0.1



new_data = np.copy(data)
for i in range(len(X)):
    x = X[i]
    y = Y[i]
    v = V[i]
    print '\n',i,x,y,v

    new_data = goodPSF.plant(x,y,v,np.copy(new_data),addNoise=True,useLinePSF=False,returnModel=False,gain = gain)



likelihoods = []
likelihoods_small = []
bRatios = []
for i in range(len(X)):
    x = X[i]
    y = Y[i]
    v = V[i]
    print '\n',i,x,y,v

    #new_data = goodPSF.plant(x,y,v,np.copy(data),addNoise=True,useLinePSF=False,returnModel=False,gain = gain)


    lsqf = opti.leastsq(resid,(x,y,v),args=(new_data-header['BGMEAN'],fitWidths[0]),maxfev=1000,ftol=ftol)
    fitPars = lsqf[0]
    l_big = likelihood(fitPars,new_data,header['BGMEAN'],boxWidth = fitWidths[0])


    lsqf = opti.leastsq(resid,(x,y,v),args=(new_data-header['BGMEAN'],fitWidths[1]),maxfev=1000,ftol=ftol)
    fitPars = lsqf[0]
    l_small = likelihood(fitPars,new_data,header['BGMEAN'],boxWidth = fitWidths[1])
    print 'planted',fitPars,l_big,l_small
    bRatios.append(fitPars[2]/v)
    likelihoods.append(l_big)
    likelihoods_small.append(l_small)

    """
    fitter = MCMCfit.MCMCfitter(goodPSF,new_data-header['BGMEAN'])
    fitter.fitWithModelPSF(x,y,m_in=v,fitWidth=8, nWalkers=6, nBurn=5, nStep=5, bg=header['BGMEAN'], useLinePSF=False, verbose=False,useErrorMap=True)
    (fitPars,fitRange)=fitter.fitResults(0.67)
    bRatios.append(fitPars[2]/v)
    likelihoods.append(fitPars[3])

    print fitPars
    removed = goodPSF.remove(fitPars[0],fitPars[1],fitPars[2],new_data,useLinePSF=False)
    fits.writeto('rem.fits',removed,clobber = True)
    fits.writeto('plant.fits',new_data,clobber = True)
    exit()
    """
bRatios = np.array(bRatios)
likelihoods = np.array(likelihoods)
likelihoods_small = np.array(likelihoods_small)



realLikelihoods = []
realVs = []
realLikelihoods_small = []
realVs_small = []
for i in range(len(lowSources)):
    x = lowSources[i][0]-1
    y = lowSources[i][1]-1
    v = lowSources[i][3]/np.sum(goodPSF.psf)
    print '\n',i+1,'/',len(lowSources),x,y,v


    lsqf = opti.leastsq(resid,(x,y,v),args=(data-header['BGMEAN'],fitWidths[0]),maxfev=1000,ftol=ftol)
    fitPars = lsqf[0]
    l_big = likelihood(fitPars,data,header['BGMEAN'],boxWidth = fitWidths[0])
    realVs.append(fitPars[2])
    realLikelihoods.append(l_big)

    lsqf = opti.leastsq(resid,(x,y,v),args=(data-header['BGMEAN'],fitWidths[1]),maxfev=1000,ftol=ftol)
    fitPars = lsqf[0]
    l_small = likelihood(fitPars,data,header['BGMEAN'],boxWidth = fitWidths[1])

    print 'real',fitPars,l_big,l_small
    realVs_small.append(fitPars[2])
    realLikelihoods_small.append(l_small)

realLikelihoods = np.array(realLikelihoods)
realVs = np.array(realVs)


fn =fits_fn.replace('.fits','.psfLikelihoods')
with open(fn,'w+') as han:
    pickle.dump([realLikelihoods,realVs,realLikelihoods_small,realVs_small,likelihoods,likelihoods_small,X,Y,V],han)
fits.writeto(fits_fn.replace('.fits','_planted.fits'),new_data,clobber = True)

fig = pyl.figure(1)
sp = fig.add_subplot(221)
pyl.scatter(V,likelihoods)
pyl.scatter(realVs,realLikelihoods,c='r')
pyl.ylabel('likelihood')
fig.add_subplot(223,sharex = sp)
pyl.scatter(V,bRatios)
pyl.xlabel('brightness')
pyl.ylabel('b ratio')
fig.add_subplot(222,sharey = sp)
pyl.scatter(bRatios,likelihoods)
pyl.xlabel('b ratio')
pyl.show()
