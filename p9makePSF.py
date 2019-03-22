import numpy as np, pylab as pyl, scipy as sci
from trippy import psf, psfStarChooser, MCMCfit, scamp
from astropy.io import fits
from stsci import numdisplay
from astropy.visualization import interval
import time
from scipy import optimize as opti
import pickle
from paths import *
import sys,glob,os

def resid(p,cutout,boxWidth = 7,verbose=False):
    (x,y,m) = p
    xt,yt = int(x),int(y)
    (a,b)=cutout.shape
    res = goodPSF.remove(x,y,m,cutout,useLinePSF=False)[yt-boxWidth:yt+boxWidth+1,xt-boxWidth:xt+boxWidth+1]
    if verbose:
        print(np.sum(res**2)**0.5,x,y,m)

    return np.array(res).astype('float').reshape((boxWidth*2+1)**2)

def likelihood(p,cutout,bg,boxWidth = 7,verbose=False):
    (x,y,m) = p
    res = resid(p,cutout-bg,boxWidth = boxWidth,verbose=verbose)
    xt,yt = int(x),int(y)
    ue2 = np.abs(cutout[yt-boxWidth:yt+boxWidth+1,xt-boxWidth:xt+boxWidth+1].reshape((boxWidth*2+1)**2))
    return -0.5*np.sum(res**2/ue2)



fits_fn = '/media/fraserw/Thumber/DEC2018_also/02530/HSC-R2/corr/CORR-0153954-083.fits'
if len(sys.argv)>1:
    files = [sys.argv[1]]

else:
    files = glob.glob(sourceDir+'/*.fits')
    files.sort()

    overwrite = False

for ff,fits_fn in enumerate(files):
    chip = fits_fn.split('-')[-1].split('.')[0]
    if not (chip in ['034','056','058','072','083']):
        continue


    fn = fits_fn.replace('/corr/','/corr/sexSaves/').replace('.fits','.cat')

    savesPath = sourceDir+'psfStars/'
    psfStars_fn = savesPath+fits_fn.replace('.fits','.psfStars').split('/')[-1]

    psf_fn = psfStars_fn.replace('.psfStars','.psf.fits')
    print(psf_fn)

    if not overwrite:
        if os.path.isfile(psf_fn):
            print('Not overwriting.')
            continue

    with open(psfStars_fn) as han:
        dat = han.readlines()
    FWHM = float(dat[0].split()[0])
    print(FWHM)

    sources = []
    for i in range(len(dat)):
        s = dat[i].split()
        sources.append([float(s[0]),float(s[1]),float(s[2]),float(s[3])])
    sources = np.array(sources)


    with fits.open(fits_fn) as han:
        data = han[1].data
        header = han[0].header
    apNum = apertures[round(FWHM*1)]
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


    genPSF = True
    if genPSF:
        starChooser = psfStarChooser.starChooser(data,
                                                 sources[:,0],sources[:,1],
                                                 sources[:,3],sources[:,2])
        (goodFits,goodMeds,goodSTDs) = starChooser(30,1,noVisualSelection=True,autoTrim=False,bgRadius = 28)

        psfWidth = int(5*goodMeds[0])
        goodPSF=psf.modelPSF(np.arange(psfWidth*2+1),np.arange(psfWidth*2+1), alpha=goodMeds[2],beta=goodMeds[3],repFact=10)
        goodPSF.genLookupTable(data,goodFits[:,4],goodFits[:,5],verbose=False,bgRadius=4*goodMeds[0])
        fwhm=goodPSF.FWHM()

        #(z1,z2)=numdisplay.zscale.zscale(goodPSF.lookupTable)
        #normer=interval.ManualInterval(z1,z2)

        print('PSF has FWHM={:.2f} pixels.'.format(fwhm))

        goodPSF.psfStore(psf_fn)
    else:
        goodPSF = psf.modelPSF(restore=fits_fn.replace('.fits','.psf.fits'))
