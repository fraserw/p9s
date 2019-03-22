import numpy as np, pylab as pyl, scipy as sci
from trippy import psf, psfStarChooser, MCMCfit, scamp
from astropy.io import fits
from stsci import numdisplay
from astropy.visualization import interval
import time
from scipy import optimize as opti, interpolate as interp
import pickle
from paths import *
import sys,glob

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


def planter(fits_fn = '/media/fraserw/rocketdata/DEC2018/02530/HSC-R2/corr/CORR-0154200-083.fits', saveToRocket=False):

    fn = fits_fn.replace('/corr/','/corr/sexSaves/').replace('.fits','.cat')

    if saveToRocket:
        savesPath = '/media/fraserw/rocketdata/scratch/psfStars_scratch/'
    else:
        savesPath = sourceDir+'psfStars/'
    psfStars_fn = savesPath+fits_fn.replace('.fits','.psfStars').split('/')[-1]


    with fits.open(fits_fn) as han:
        data = han[1].data
        header = han[0].header
    data += header['BGMEAN']
    gain = header['MEDGAIN']

    #measuring faint sources for training set generation
    goodPSF = psf.modelPSF(restore=fits_fn.replace('/corr/','/corr/psfStars/').replace('.fits','.psf.fits'))
    fwhm = goodPSF.FWHM()

    pWidth = int( (goodPSF.psf.shape[0]-1)/2 )

    #planting point sources for training set generation
    (A,B) = data.shape

    #for planting faint want 5<snr<60.0
    new_data = np.copy(data)*0.0
    x,y,v = 100.0,100.0,1.0
    new_data = goodPSF.plant(x,y,v,np.copy(new_data),addNoise=False,useLinePSF=False,returnModel=False,gain = gain,plantBoxWidth=pWidth)
    flux = np.sum(new_data)
    bg_flux = np.pi*fwhm**2*header['BGMEAN']


    multis = 10.0**np.linspace(-4,4,5000.0)
    snr = multis*flux*(multis*flux+bg_flux)**-0.5

    f = interp.interp1d(snr,multis)
    low_multi = f(3.0)
    high_multi = f(70.0)
    print(low_multi,high_multi)



    nplant_f = 500
    X_f = sci.rand(nplant_f)*B#(B-pWidth*2)+pWidth
    Y_f = sci.rand(nplant_f)*A
    #up to 2 gives SNR~60 or less
    V_f = sci.rand(nplant_f)*(high_multi-low_multi)+low_multi


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
        han.writeto(savesPath+fits_fn.split('/')[-1].replace('.fits','_faint_planted.fits'), overwrite = True)

    with open(savesPath+fits_fn.split('/')[-1].replace('.fits','_faint_planted.coords'),'w+') as han:
        for i in range(len(X_f)):
            han.write("{} {} {}\n".format(X_f[i],Y_f[i],V_f[i]))


    #for planting bright sources with 60.0<SNR<1000
    low_multi = f(50.0)
    high_multi = f(1000.0)

    print(low_multi,high_multi)
    nplant_b = 300
    X_b = sci.rand(nplant_b)*B
    Y_b = sci.rand(nplant_b)*A
    V_b = sci.rand(nplant_b)*(high_multi-low_multi)+low_multi


    new_data = np.copy(data)
    for i in range(len(X_b)):
        x = X_b[i]
        y = Y_b[i]
        v = V_b[i]
        print(i,'b',x,y,v)

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
        han.writeto(savesPath+fits_fn.split('/')[-1].replace('.fits','_bright_planted.fits'), overwrite = True)

    with open(savesPath+fits_fn.split('/')[-1].replace('.fits','_bright_planted.coords'),'w+') as han:
        for i in range(len(X_b)):
            han.write("{} {} {}\n".format(X_b[i],Y_b[i],V_b[i]))

    return savesPath

    #planting binaries
    nplant_bin = 300
    low_multi = f(4.0)
    high_multi = f(70.0)

    X_bin = sci.rand(nplant_bin)*B
    Y_bin = sci.rand(nplant_bin)*A
    #up to 300 for the PSF stars
    V_bin = sci.rand(nplant_bin)*(high_multi-low_multi)+low_multi
    X_sec = np.clip(X_bin + sci.rand(nplant_bin)*10.0-5.0,1,B-1)
    Y_sec = np.clip(Y_bin + sci.rand(nplant_bin)*10.0-5.0,1,A-1)
    V_sec = V_bin*sci.rand(nplant_bin)*0.8
    w = np.where(V_sec<0.05)
    V_sec[w] = 0.05


    new_data = np.copy(data)
    for i in range(len(X_bin)):
        x = X_bin[i]
        y = Y_bin[i]
        v = V_bin[i]
        x2 = X_sec[i]
        y2 = Y_sec[i]
        v2 = V_sec[i]
        print(i,x,y,v,x2,y2,v2)


        if x<pWidth or x>B-pWidth or y<pWidth or y>A-pWidth:
            new_data = goodPSF.plant(x,y,v,np.copy(new_data),addNoise=True,useLinePSF=False,returnModel=False,gain = gain,plantBoxWidth=pWidth)
        else:
            a = int(y-pWidth)
            b = int(y+pWidth+1)
            c = int(x-pWidth)
            d = int(x+pWidth+1)
            new_data[a:b,c:d] = goodPSF.plant(x-int(x)+pWidth,y-int(y)+pWidth,v,np.copy(new_data[a:b,c:d]),addNoise=True,useLinePSF=False,returnModel=False,gain = gain,plantBoxWidth=pWidth)
        if x2<pWidth or x2>B-pWidth or y2<pWidth or y2>A-pWidth:
            new_data = goodPSF.plant(x2,y2,v2,np.copy(new_data),addNoise=True,useLinePSF=False,returnModel=False,gain = gain,plantBoxWidth=pWidth)
        else:
            a = int(y2-pWidth)
            b = int(y2+pWidth+1)
            c = int(x2-pWidth)
            d = int(x2+pWidth+1)
            new_data[a:b,c:d] = goodPSF.plant(x2-int(x2)+pWidth,y2-int(y2)+pWidth,v2,np.copy(new_data[a:b,c:d]),addNoise=True,useLinePSF=False,returnModel=False,gain = gain,plantBoxWidth=pWidth)

        #new_data = goodPSF.plant(x,y,v,np.copy(new_data),addNoise=True,useLinePSF=False,returnModel=False,gain = gain)
        new_data = goodPSF.plant(x2,y2,v2,np.copy(new_data),addNoise=True,useLinePSF=False,returnModel=False,gain = gain)


    with fits.open(fits_fn) as han:
        for i in range(len(han[1].data)):
            han[1].data[i] = new_data[i]
        han[0].header['FWHMplant'] = fwhm
        han.writeto(savesPath+fits_fn.split('/')[-1].replace('.fits','_binary_planted.fits'), overwrite = True)

    with open(savesPath+fits_fn.split('/')[-1].replace('.fits','_binary_planted.coords'),'w+') as han:
        for i in range(len(X_bin)):
            han.write("{} {} {} {} {} {}\n".format(X_bin[i],Y_bin[i],V_bin[i],X_sec[i],Y_sec[i],V_sec[i]))

    return savesPath


if __name__ == "__main__":
    fits_fn = '/media/fraserw/rocketdata/DEC2018/02530/HSC-R2/corr/CORR-0154200-083.fits'
    if len(sys.argv)>1:
        files = [sys.argv[1]]

    else:
        files = glob.glob(sourceDir+'/*.fits')
        files.sort()

    for ff,fits_fn in enumerate(files):
        chip = fits_fn.split('-')[-1].split('.')[0]
        if not (chip in ['034','056','058','072','083']):
            continue
        planter(fits_fn)




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
    """
