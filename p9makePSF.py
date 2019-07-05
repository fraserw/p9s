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
import multiprocessing as multip


def getEdgeBG(lu):
    """
    Use the 2 edge pixel strips around the outside of the PSF
    to estimate the background standard deviation of the non-source contaminated
    regions of the PSF lookup table.
    """

    a = lu[:2,:]
    b = lu[-2:,:]
    c = lu[:,:2]
    d = lu[:,-2:]
    (ma,sa) = (np.median(a),np.std(a))
    (mb,sb) = (np.median(b),np.std(b))
    (mc,sc) = (np.median(c),np.std(c))
    (md,sd) = (np.median(d),np.std(d))
    #s = np.array([sa,sb,sc,sd])
    #args = np.argsort(s)
    #print(s[args])
    #return s[args[1]]
    return np.min(np.array([sa,sb,sc,sd]))




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


def makePSF(fits_fn,overwrite = True, minStars = 8, minSNRcheck = 25 ,checkMinDist = True):

    fn = fits_fn.replace('/corr/','/corr/sexSaves/').replace('.fits','.cat')
    psfStars_fn = savesPath+fits_fn.replace('.fits','.psfStars').split('/')[-1]

    psf_fn = psfStars_fn.replace('.psfStars','.psf.fits')
    print(psf_fn)

    if not overwrite:
        if os.path.isfile(psf_fn):
            print('Not overwriting.')
            return

    with open(psfStars_fn) as han:
        dat = han.readlines()
    FWHM = float(dat[0].split()[0])
    print('FWHM',FWHM)

    sources = []
    for i in range(1,len(dat)):
        s = dat[i].split()
        sources.append([float(s[0]),float(s[1]),float(s[2]),float(s[3])])
    sources = np.array(sources)

    #print(fits_fn)

    with fits.open(fits_fn) as han:
        data = han[1].data
        header = han[0].header
    apNum = apertures[round(FWHM*1)]
    data += header['BGMEAN']
    gain = header['MEDGAIN']

    if checkMinDist:
        goodSources = []
        while len(goodSources)<min(minStars,len(sources)) and minSNRcheck<50:
            catalog = scamp.getCatalog(fn,paramFile='sextract.param')
            w = np.where((catalog['X_IMAGE']>5) & (catalog['X_IMAGE']<2045) & (catalog['Y_IMAGE']>5) & (catalog['Y_IMAGE']<4173) \
                    &  ((catalog['FLUX_APER(9)'][:,apNum]/catalog['FLUXERR_APER(9)'][:,apNum])>minSNRcheck) \
                    &  ((catalog['FLUX_APER(9)'][:,apNum]/catalog['FLUXERR_APER(9)'][:,apNum])<500) & (catalog['FWHM_IMAGE']>1.5))

            lowSources = np.zeros((len(w[0]),4)).astype('float64')
            lowSources[:,0] = catalog['X_IMAGE'][w]
            lowSources[:,1] = catalog['Y_IMAGE'][w]
            lowSources[:,2] = catalog['FLUXERR_APER(9)'][:,apNum][w]
            lowSources[:,3] = catalog['FLUX_APER(9)'][:,apNum][w]

            #snrs = (catalog['FLUX_APER(9)'][:,apNum]/catalog['FLUXERR_APER(9)'][:,apNum])[w]

            goodSources = []
            for i in range(len(sources)):
                #d1 = ((sources[i,0]-sources[:,0])**2 + (sources[i,1]-sources[:,1])**2)**0.5
                #w1 = np.where(d1<5*FWHM)
                d2 = ((sources[i,0]-lowSources[:,0])**2 + (sources[i,1]-lowSources[:,1])**2)**0.5
                w2 = np.where(d2<4*FWHM)
                #args = np.argsort(d2)
                #print(i,sources[i,0],sources[i,1],lowSources[args[0],0],lowSources[args[0],1],snrs[args[0]])
                #print(len(w2[0]),np.min(d2))
                if len(w2[0])==1:
                    goodSources.append(sources[i])
            if (len(goodSources)<min(minStars,len(sources)) and minSNRcheck<50):
                minSNRcheck+=5
        if len(goodSources)>=5:
            sources = np.array(goodSources)
        else:
            print('Only {} isolated sources found!'.format(len(goodSources)))
            psf_fn = psf_fn.replace('.psf.','..psf_inspect.')

    #for i in range(len(sources)):
    #    print(sources[i,:2])
    #exit()
    div = max(int(len(sources)/40),1) #taking at most 40 sources
    print('Using {} sources after trimming with minSNRcheck={} with div={}.'.format(len(sources[::div,0]),minSNRcheck,div))

    if FWHM<15:
        initAlpha = 3.0
        initBeta = 3.0
    else:
        initAlpha = 5.0
        initBeta = 3.0
    starChooser = psfStarChooser.starChooser(data,
                                             sources[::div,0],sources[::div,1],
                                             sources[::div,3],sources[::div,2])
    #print(max(28,int(4*FWHM)))
    #exit()
    #wid = max(30,int(6*FWHM))
    (goodFits,goodMeds,goodSTDs) = starChooser(30,1,noVisualSelection=False,autoTrim=False,bgRadius = 28,quickFit = True)#, initAlpha=initAlpha, initBeta = initBeta)##ftol = 1.e-7)
    psfWidth = int(5*goodMeds[0]) #half width

    goodPSF = psf.modelPSF(np.arange(psfWidth*2+1),np.arange(psfWidth*2+1), alpha=goodMeds[2],beta=goodMeds[3],repFact=10)
    goodPSF.genLookupTable(data,goodFits[:,4],goodFits[:,5],verbose=False,bgRadius=6*goodMeds[0])

    fwhm = goodPSF.FWHM(fromMoffatProfile = True)
    print('PSF has moffat FWHM={:.2f} pixels.'.format(fwhm))

    #goodPSF.psfStore('junk.psf.fits')



    #psf residual fix from testPSF.py
    lu = goodPSF.lookupTable

    estd = getEdgeBG(lu)

    (A,B) = lu.shape
    (y,x) = np.meshgrid(np.arange(A),np.arange(B))
    r = (((y-A/2.+0.5)**2 + (x-B/2.+0.5)**2)**0.5)/10.0


    #w = np.where(r>4.0*fwhm)
    #std = np.nanstd(lu[w])
    #print(estd,std,'**')
    #if std>4*estd:
    #    std = estd
    std = estd

    rand = sci.randn(A,B)*std
    W = np.where((lu>3.0*std) & (r>4.0*fwhm))
    goodPSF.lookupTable[W] = rand[W]

    goodPSF.psfStore(psf_fn, psfV2 = True)

    with fits.open(psf_fn) as han:
        header = han[0].header
        header.set('fwhm',fwhm)
        han[0].header = header
        han.writeto(psf_fn,overwrite = True)

    return FWHM



overwrite = True
fits_fn = '/media/fraserw/Thumber/DEC2018_also/02530/HSC-R2/corr/CORR-0154074-034.fits'
if len(sys.argv)>1:
    files = [sys.argv[1]]

else:
    files = glob.glob('/media/fraserw/Hammer/DEC2018/*/HSC-R2/corr/CORR-???????-???.fits')+glob.glob('/media/fraserw/Thumber/DEC2018_also/*/HSC-R2/corr/CORR-???????-???.fits')
    #files = glob.glob(sourceDir+'/*.fits')
    files.sort()

    overwrite = False


Files = []

for ff,fits_fn in enumerate(files):
    chip = fits_fn.split('-')[-1].split('.')[0]
    #if not (chip in ['034','056','058','072','083']):
    #    continue
    Files.append(fits_fn)

files = Files[:]


#savesPath = sourceDir+'psfStars/'
savesPath = '/media/fraserw/rocketdata/DEC2018/psfStars/'

numCores = 1
if numCores == 1:
    for ff,fits_fn in enumerate(files):

        fn = fits_fn.replace('/corr/','/corr/sexSaves/').replace('.fits','.cat')
        makePSF(fits_fn, overwrite=overwrite, minSNRcheck = 5, minStars = 20)

else:
    q = multip.Queue()
    i = 0
    while i < len(files)+1:

        processes = []
        for j in range(numCores):
            if i+j == len(files):
                break
            fits_fn = files[i+j]
            fn = fits_fn.replace('/corr/','/corr/sexSaves/').replace('.fits','.cat')


            processes.append(multip.Process(target=makePSF,args=(fits_fn,overwrite)))
            processes[j].start()

        for j in range(len(processes)):
            processes[j].join()

        i+=numCores

"""
    psfStars_fn = savesPath+fits_fn.replace('.fits','.psfStars').split('/')[-1]

    psf_fn = psfStars_fn.replace('.psfStars','.psf= rand[W]#.fits')
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
            &  ((catalog['FLUX_APER(9)'][:,apNum]/catalog['FLUXERR_APER(9)'][:,apNum])>5) \
            &  ((catalog['FLUX_APER(9)'][:,apNum]/catalog['FLUXERR_APER(9)'][:,apNum])<50) & (catalog['FWHM_IMAGE']>1.5))
    lowSources = np.zeros((len(w[0]),4)).astype('float64')
    lowSources[:,0] = catalog['X_IMAGE'][w]
    lowSources[:,1] = catalog['Y_IMAGE'][w]
    lowSources[:,2] = catalog['FLUXERR_APER(9)'][:,apNum][w]
    lowSources[:,3] = catalog['FLUX_APER(9)'][:,apNum][w]


    genPSF = True
    if genPSF:
        div = max(int(len(sources)/40.0),1) #taking at least 40 sources
        starChooser = psfStarChooser.starChooser(data,
                                                 sources[::div,0],sources[::div,1],
                                                 sources[::div,3],sources[::div,2])
        (goodFits,goodMeds,goodSTDs) = starChooser(30,1,noVisualSelection=True,autoTrim=False,bgRadius = 28,ftol = 1.e-6)

        psfWidth = int(8*goodMeds[0])
        goodPSF=psf.modelPSF(np.arange(psfWidth*2+1),np.arange(psfWidth*2+1), alpha=goodMeds[2],beta=goodMeds[3],repFact=10)
        goodPSF.genLookupTable(data,goodFits[:,4],goodFits[:,5],verbose=False,bgRadius=6*goodMeds[0])

        fwhm=goodPSF.FWHM()

        #(z1,z2)=numdisplay.zscale.zscale(goodPSF.lookupTable)
        #normer=interval.ManualInterval(z1,z2)

        print('PSF has FWHM={:.2f} pixels.'.format(fwhm))

        goodPSF.psfStore(psf_fn)
    else:
        goodPSF = psf.modelPSF(restore=fits_fn.replace('.fits','.psf.fits'))
"""
