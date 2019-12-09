import numpy as np, pylab as pyl, scipy as sci
from trippy import psf, psfStarChooser, MCMCfit, scamp, tzscale
from astropy.io import fits
from astropy.visualization import interval
import time
from scipy import optimize as opti, interpolate as interp, stats
import pickle
from paths import *
import sys,glob,os
from p9extract import runSex

"""
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
"""

def eff(m,A=0.9,m50=24.5,g=0.2):
    return 0.5*A*(1.0-np.tanh((m-m50)/g))

def resid(p,m,e,w):
    (A,m50,g) = p
    E = eff(m,A,m50,g)
    print(np.sqrt(np.sum((E-e)**2)/len(E)),A,m50,g)
    return (E-e)*w



def findEff(fits_fn = '/media/fraserw/rocketdata/DEC2018/02530/HSC-R2/corr/CORR-0154200-083.fits', saveToRocket=True, foundThreshold = 1.0,overwrite = False):

    #setting the various input and output filenames
    fn = fits_fn.split('/')[-1].replace('.fits','_planted.fits')

    if saveToRocket:
        savesPath = '/media/fraserw/rocketdata/scratch/psfStars_scratch/'
    else:
        savesPath = sourceDir+'psfStars/'
    psfStars_fn = savesPath+fits_fn.replace('.fits','.psfStars').split('/')[-1]
    plant_fn = savesPath+fn

    sex_save_file = plant_fn.replace('.fits','.sex_save')
    mask_file = '/home/fraserw/idl_progs/hscp9/sextract/mask'+str(chip).zfill(3)+'.fits'


    eff_fn = plant_fn.replace('_planted.fits','.eff')
    if os.path.isfile(eff_fn) and not overwrite:
        print('    Already done. Skipping.\n')
        return
    elif overwrite:
        print('    Overwriting the extant eff file.\n')

    psf_fn = '/media/fraserw/rocketdata/DEC2018/psfStars/'+fits_fn.split('/')[-1].replace('.fits','.psf.fits')


    #filenames set

    with fits.open(fits_fn) as han:
        data = han[1].data
        header = han[0].header
    data += header['BGMEAN']
    gain = header['MEDGAIN']


    if not os.path.isfile(psf_fn):
        if os.path.isfile(psf_fn.replace('-'+chip,'-034')):
            psf_fn = psf_fn.replace('-'+chip,'-034')
        elif os.path.isfile(psf_fn.replace('-'+chip,'-056')):
            psf_fn = psf_fn.replace('-'+chip,'-056')
        elif os.path.isfile(psf_fn.replace('-'+chip,'-058')):
            psf_fn = psf_fn.replace('-'+chip,'-058')
        elif os.path.isfile(psf_fn.replace('-'+chip,'-072')):
            psf_fn = psf_fn.replace('-'+chip,'-072')
        elif os.path.isfile(psf_fn.replace('-'+chip,'-083')):
            psf_fn = psf_fn.replace('-'+chip,'-083')

    #measuring faint sources for training set generation
    goodPSF = psf.modelPSF(restore=psf_fn)
    fwhm = goodPSF.FWHM()

    pWidth = int( (goodPSF.psf.shape[0]-1)/2 )

    #planting point sources for training set generation
    (A,B) = data.shape

    #for planting faint want 5<snr<60.0
    new_data = np.copy(data)*0.0
    x,y,v = 100.0,100.0,1.0
    new_data = goodPSF.plant(x,y,v,np.copy(new_data),addNoise=False,useLinePSF=False,returnModel=False,gain = gain,plantBoxWidth=pWidth)
    #fits.writeto('junk.fits',new_data+sci.randn(A,B)*header['BGMEAN']**0.5+header['BGMEAN'],overwrite=True)

    unit_signal = np.sum(new_data)
    bg_signal = np.pi*fwhm**2*header['BGMEAN']
    #print(bg_signal,unit_signal)
    #print(header['EXPTIME'],header['MAGZERO'])
    #exit()

    ###initial low fidelity fit.

    #desired range of magnitudes
    mag_lim = np.array([20.0,26.5])
    signal_lims = header['EXPTIME']*10.0**(-0.4*(mag_lim-header['MAGZERO']))
    multi_lim = signal_lims/unit_signal
    mag_lim_f = np.array([24.0,26.5])
    signal_lims_f = header['EXPTIME']*10.0**(-0.4*(mag_lim_f-header['MAGZERO']))
    multi_lim_f = signal_lims_f/unit_signal

    dmag = 0.25
    mag_bins = np.arange(mag_lim[0]-dmag,mag_lim[1]+dmag,dmag)
    bins = mag_bins*0.0
    plant_bins = mag_bins*0.0

    n_rep = 4
    nplant = 100#500
    nplant_f = 0#200
    for n_run in range(n_rep):
        X_f = np.concatenate([sci.rand(nplant)*B,sci.rand(nplant_f)*B])
        Y_f = np.concatenate([sci.rand(nplant)*A,sci.rand(nplant_f)*A])
        #up to 2 gives SNR~60 or less
        V_f = np.concatenate([sci.rand(nplant)*(multi_lim[0]-multi_lim[1])+multi_lim[1], sci.rand(nplant_f)*(multi_lim_f[0]-multi_lim_f[1])+multi_lim_f[1]])

        flux_f = V_f*unit_signal
        mag_f = header['MAGZERO']-2.5*np.log10(flux_f/header['EXPTIME'])

        new_data = np.copy(data)
        for i in range(len(X_f)):
            x = X_f[i]
            y = Y_f[i]
            v = V_f[i]
            print(i,x,y,v)
            if x<pWidth or x>B-pWidth or y<pWidth or y>A-pWidth:
                new_data = goodPSF.plant(x,y,v,np.copy(new_data),addNoise=True,useLinePSF=False,returnModel=False,gain = gain,plantBoxWidth=pWidth)
            else:
                a = int(y-pWidth)
                b = int(y+pWidth+1)
                c = int(x-pWidth)
                d = int(x+pWidth+1)
                new_data[a:b,c:d] = goodPSF.plant(x-int(x)+pWidth,y-int(y)+pWidth,v,np.copy(new_data[a:b,c:d]),addNoise=True,useLinePSF=False,returnModel=False,gain = gain,plantBoxWidth=pWidth)


        with fits.open(fits_fn) as han:
            for i in range(len(han[1].data)):
                han[1].data[i] = new_data[i]
            han[0].header['FWHMplant'] = fwhm
            han.writeto(savesPath+fits_fn.split('/')[-1].replace('.fits','_planted.fits'), overwrite = True)

        catalog = runSex(plant_fn,fn,int(chip),mask_file,savesPath,showProgress = False, verbose = True, includeImageMask = True,kron_cut = -1000.0,runSextractor = True,snr_lim=3)
        found = np.ones(len(X_f))
        for ii in range(len(X_f)):
            dist = ((catalog.x - X_f[ii]+0.5)**2 + (catalog.y - Y_f[ii]+0.5)**2)**0.5
            if np.min(dist)>foundThreshold:
                found[ii] = 0.0

        k = ((mag_f-mag_bins[0])/dmag).astype('int')
        for ind,ii in enumerate(k):
            #print(ind,ii)
            plant_bins[ii]+=1.0
            if found[ind]:
                bins[ii]+=1.0
    bins = np.array(bins)
    plant_bins = np.array(plant_bins)
    mag_eff = bins/plant_bins
    w = np.where(np.isnan(mag_eff))
    mag_eff[w] = 0.0
    mag_bins +=dmag/2.0

    w = np.where(mag_bins<21)
    Ao = np.median(mag_eff[w])
    k = np.sum(np.greater(mag_eff,Ao/2.0))
    m50o = mag_bins[k]
    go = 0.5

    x = plant_bins-bins

    uncert = (stats.poisson.ppf(.82,x)-stats.poisson.ppf(.15,x))/plant_bins
    w = np.where(plant_bins==0.0)
    uncert[w] = np.nanmax(uncert)
    uncert = np.clip(uncert,0.05,1.0)
    weights = 1.0/uncert
    #weights/=np.sum(weights)

    (A_lf,m50,g) = opti.leastsq(resid,(Ao,m50o,go),(mag_bins,mag_eff,weights))[0]



    ###higher fidelity fit
    #desired range of magnitudes
    mag_lim = np.array([round(m50,1)-2.5,round(m50,1)+1.0])
    signal_lims = header['EXPTIME']*10.0**(-0.4*(mag_lim-header['MAGZERO']))
    multi_lim = signal_lims/unit_signal
    mag_lim_f = np.array([round(m50,1)-0.55,round(m50,1)+0.5])
    signal_lims_f = header['EXPTIME']*10.0**(-0.4*(mag_lim_f-header['MAGZERO']))
    multi_lim_f = signal_lims_f/unit_signal


    #0.828 23.252 0.472
    dmag = 0.2
    mag_bins = np.arange(mag_lim[0]-dmag,mag_lim[1]+dmag,dmag)
    bins = mag_bins*0.0
    plant_bins = mag_bins*0.0

    n_rep = 4
    nplant = 300#500
    nplant_f = 100#200
    for n_run in range(n_rep):
        X_f = np.concatenate([sci.rand(nplant)*B,sci.rand(nplant_f)*B])
        Y_f = np.concatenate([sci.rand(nplant)*A,sci.rand(nplant_f)*A])
        #up to 2 gives SNR~60 or less
        V_f = np.concatenate([sci.rand(nplant)*(multi_lim[0]-multi_lim[1])+multi_lim[1], sci.rand(nplant_f)*(multi_lim_f[0]-multi_lim_f[1])+multi_lim_f[1]])

        flux_f = V_f*unit_signal
        mag_f = header['MAGZERO']-2.5*np.log10(flux_f/header['EXPTIME'])

        new_data = np.copy(data)
        for i in range(len(X_f)):
            x = X_f[i]
            y = Y_f[i]
            v = V_f[i]
            print(i,x,y,v)
            if x<pWidth or x>B-pWidth or y<pWidth or y>A-pWidth:
                new_data = goodPSF.plant(x,y,v,np.copy(new_data),addNoise=True,useLinePSF=False,returnModel=False,gain = gain,plantBoxWidth=pWidth)
            else:
                a = int(y-pWidth)
                b = int(y+pWidth+1)
                c = int(x-pWidth)
                d = int(x+pWidth+1)
                new_data[a:b,c:d] = goodPSF.plant(x-int(x)+pWidth,y-int(y)+pWidth,v,np.copy(new_data[a:b,c:d]),addNoise=True,useLinePSF=False,returnModel=False,gain = gain,plantBoxWidth=pWidth)


        with fits.open(fits_fn) as han:
            for i in range(len(han[1].data)):
                han[1].data[i] = new_data[i]
            han[0].header['FWHMplant'] = fwhm
            han.writeto(savesPath+fits_fn.split('/')[-1].replace('.fits','_planted.fits'), overwrite = True)

        catalog = runSex(plant_fn,fn,int(chip),mask_file,savesPath,showProgress = False, verbose = True, includeImageMask = True,kron_cut = -1000.0,runSextractor = True,snr_lim=3)

        found = np.ones(len(X_f))
        for ii in range(len(X_f)):
            dist = ((catalog.x - X_f[ii]+0.5)**2 + (catalog.y - Y_f[ii]+0.5)**2)**0.5
            if np.min(dist)>foundThreshold:
                found[ii] = 0.0

        k = ((mag_f-mag_bins[0])/dmag).astype('int')
        for ind,ii in enumerate(k):
            #print(ind,ii)
            plant_bins[ii]+=1.0
            if found[ind]:
                bins[ii]+=1.0
                print(X_f[ii],mag_f[ii])
    bins = np.array(bins)
    plant_bins = np.array(plant_bins)
    mag_eff = bins/plant_bins
    w = np.where(np.isnan(mag_eff))
    mag_eff[w] = 0.0
    mag_bins +=dmag/2.0

    Ao = np.median(mag_eff[:5])
    k = np.sum(np.greater(mag_eff,Ao/2.0))
    m50o = mag_bins[k]
    go = 0.5

    x = plant_bins-bins

    uncert = (stats.poisson.ppf(.82,x)-stats.poisson.ppf(.15,x))/plant_bins
    w = np.where(plant_bins==0.0)
    uncert[w] = np.nanmax(uncert)
    uncert = np.clip(uncert,0.05,1.0)
    weights = 1.0/uncert
    #weights/=np.sum(weights)

    (A,m50,g) = opti.leastsq(resid,(Ao,m50o,go),(mag_bins,mag_eff,weights))[0]

    fig = pyl.figure('eff')
    pyl.scatter(mag_bins[1:],mag_eff[1:])
    m = np.linspace(mag_bins[0],mag_bins[-1],100)
    pyl.plot(m,eff(m,A,m50,g))
    pyl.savefig(plant_fn.replace('_planted.fits','.png'),bbox_inches='tight')
    pyl.clf()
    """
    pyl.show()
    exit()
    """
    with open(eff_fn,'w+') as outhan:
        print('# {:.3f} {:.3f} {:.3f}'.format(A,m50,g))
        print('# {:.3f} {:.3f} {:.3f}'.format(A,m50,g),file=outhan)
        for ii in range(1,len(plant_bins)):
            print('{:.3f} {:.3f} {:.3f}'.format(mag_bins[ii],plant_bins[ii],bins[ii]/plant_bins[ii] if not np.isnan(bins[ii]/plant_bins[ii]) else 0.0))
            print('{:.3f} {:.3f} {:.3f}'.format(mag_bins[ii],plant_bins[ii],bins[ii]/plant_bins[ii] if not np.isnan(bins[ii]/plant_bins[ii]) else 0.0),file=outhan)

    os.system('rm {}'.format(plant_fn))
    os.system('rm {}'.format(sex_save_file))
    os.system('rm {}'.format(savesPath+fn.replace('.fits','.cat')))
    return None

if __name__ == "__main__":
    kron_cut = -10000.5
    runSextractor = True
    overwrite = True

    """
    m = np.linspace(23,25,100)
    print(eff(m))
    pyl.plot(m,eff(m))
    pyl.show()
    exit()
    """

    fits_fn = '/media/fraserw/Thumber/DEC2018_also/02530/HSC-R2/corr/CORR-0156174-034.fits'
    if len(sys.argv)>1:
        files = [sys.argv[1]]

    else:
        files = glob.glob(sourceDir+'/*.fits')
        files.sort()

    for ff,fits_fn in enumerate(files):
        chip = fits_fn.split('-')[-1].split('.')[0]
        if not (chip in ['034','056','058','072','083']):
            continue
        print('Running {}.'.format(fits_fn))
        findEff(fits_fn,overwrite=overwrite)
        print('Finished {}.\n'.format(fits_fn))




    # old lieklihood fitting code I am keeping for prosperity
    """

        dmbg = new_data - header['BGMEAN']

        t1 = time.time()
        likelihoods = []
        likelihoods_small = []
        bRatios = []
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            v = V[i]
            print('\n',i,x,y,v)

            #new_data = goodPSF.plant(x,y,v,np.copy(data),addNoise=True,useLinePSF=False,returnModel=False,gain = gain)


            lsqf = opti.leastsq(resid,(x,y,v),args=(dmbg,fitWidths[0]),maxfev=1000,ftol=ftol)
            fitPars = lsqf[0]
            l_big = likelihood(fitPars,new_data,header['BGMEAN'],boxWidth = fitWidths[0])


            lsqf = opti.leastsq(resid,(x,y,v),args=(dmbg,fitWidths[1]),maxfev=1000,ftol=ftol)
            fitPars = lsqf[0]
            l_small = likelihood(fitPars,new_data,header['BGMEAN'],boxWidth = fitWidths[1])
            print('planted',fitPars,l_big,l_small)
            bRatios.append(fitPars[2]/v)
            likelihoods.append(l_big)
            likelihoods_small.append(l_small)


        bRatios = np.array(bRatios)
        likelihoods = np.array(likelihoods)
        likelihoods_small = np.array(likelihoods_small)


        dmbg = data - header['BGMEAN']

        realLikelihoods = []
        realVs = []
        realLikelihoods_small = []
        realVs_small = []
        realX = []
        realY=[]
        for i in range(len(lowSources)):
            x = lowSources[i][0]-1
            y = lowSources[i][1]-1
            v = lowSources[i][3]/np.sum(goodPSF.psf)
            print('\n',i+1,'/',len(lowSources),x,y,v)


            lsqf = opti.leastsq(resid,(x,y,v),args=(dmbg,fitWidths[0]),maxfev=1000,ftol=ftol)
            fitPars = lsqf[0]
            l_big = likelihood(fitPars,data,header['BGMEAN'],boxWidth = fitWidths[0])
            realVs.append(fitPars[2])
            realLikelihoods.append(l_big)
            realX.append(fitPars[0])
            realY.append(fitPars[1])

            lsqf = opti.leastsq(resid,(x,y,v),args=(dmbg,fitWidths[1]),maxfev=1000,ftol=ftol)
            fitPars = lsqf[0]
            l_small = likelihood(fitPars,data,header['BGMEAN'],boxWidth = fitWidths[1])

            print('real',fitPars,l_big,l_small)
            realVs_small.append(fitPars[2])
            realLikelihoods_small.append(l_small)


        realLikelihoods = np.array(realLikelihoods)
        realVs = np.array(realVs)
        realX = np.array(realX)
        realY = np.array(realY)


        fn =fits_fn.replace('.fits','.psfLikelihoods')
        with open(fn,'w+') as han:
            pickle.dump([realLikelihoods,realVs,realLikelihoods_small,realVs_small,realX,realY,likelihoods,likelihoods_small,X,Y,V],han)


        exit()



        fig = pyl.figure(1)
        sp = fig.add_subplot(221)
        pyl.scatter(V,likelihoods)
        pyl.scatter(realVs,realLikelihoods,c='r')(savesPath,fn) =
        pyl.ylabel('likelihood')
        fig.add_subplot(223,sharex = sp)
        pyl.scatter(V,bRatios)
        pyl.xlabel('brightness')
        pyl.ylabel('b ratio')
        fig.add_subplot(222,sharey = sp)
        pyl.scatter(bRatios,likelihoods)
        pyl.xlabel('b ratio')
        pyl.show()
    """


    """
ERROR LOG FROM LAST run
p9GetObsEff.py:190: RuntimeWarning: invalid value encountered in double_scalars
  print('{:.3f} {:.3f} {:.3f}'.format(mag_bins[ii],plant_bins[ii],bins[ii]/plant_bins[ii]))
26.550 0.000 nan
p9GetObsEff.py:191: RuntimeWarning: invalid value encountered in double_scalars
  print('{:.3f} {:.3f} {:.3f}'.format(mag_bins[ii],plant_bins[ii],bins[ii]/plant_bins[ii]),file=outhan)
26.650 0.000 nan
"""
