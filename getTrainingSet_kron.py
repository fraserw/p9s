
import scipy as sci, numpy as np, pylab as pyl, glob, os
from os import path
from paths import *
from astropy.io import fits
from astropy.visualization import interval
from astropy import wcs
from stsci import numdisplay
import multiprocessing as multip
import scamp
import pickle
import matplotlib.gridspec as gridspec
from p9extract import getMeanMagDiff, runSex
from scipy import interpolate as interp
from p9plantPSF import planter
from trippy import bgFinder

def cutter(x,y,cut,A,B,c,bg):
    (ca,cb) = cut.shape
    z = np.zeros((2*c+1,2*c+1),dtype=np.float64)+bg

    aa = max(0,y-(c+1))
    cc = max(0,x-(c+1))
    bb = min(A,y+c+1)
    dd = min(B,x+c+1)

    AA = 0
    BB = 2*c+1
    CC = 0
    DD = 2*c+1
    if aa == 0:
        BB = 2*c+1
        AA = BB-(bb-aa)
    elif bb == A:
        AA = 0
        BB = AA+(bb-aa-1)
    if cc == 0:
        DD = 2*c+1
        CC = DD-(dd-cc)
    elif dd == B:
        CC = 0
        DD = AA+(dd-cc-1)
    z[AA:BB,CC:DD] = cut
    return z

def showSources(pred,catalog,data,c=8,returnCuts = False,snr=None,appended=''):
    X = catalog['X_IMAGE']
    Y = catalog['Y_IMAGE']
    (A,B) = data.shape

    (z1,z2) = numdisplay.zscale.zscale(data,contrast = 0.5)
    normer = interval.ManualInterval(z1,z2)

    w = np.where(pred == 1)
    nsp = int(len(w[0])**0.5)
    if len(w[0])>nsp*nsp:
        nsp += 1
    if not returnCuts:
        fig = pyl.figure('good'+appended,figsize=(15,15))
        fig.subplots_adjust(hspace=0,wspace=0)
        gs = gridspec.GridSpec(nsp,nsp)

    else:
        cuts = []
        snrs = []
    for ii in range(nsp):
        for jj in range(nsp):
            if ii*nsp+jj<len(w[0]):
                x,y = int(X[w[0]][ii*nsp+jj])-1,int(Y[w[0]][ii*nsp+jj])-1
                cut = data[y-c:y+c+1,x-c:x+c+1]
                if np.min(cut.shape)<3:
                    continue
                if cut.shape[0]!=2*c+1 or cut.shape[1]!=2*c+1:
                    cut = cutter(x,y,cut,A,B,c,returnCuts)
                if returnCuts:
                    cuts.append(cut)
                    snrs.append(snr[w[0]][ii*nsp+jj])
                else:
                    try:
                        (z1,z2) = numdisplay.zscale.zscale(cut,contrast = 0.5)
                        normer = interval.ManualInterval(z1,z2)
                        Cut = normer(cut)
                        sp = pyl.subplot(gs[ii,jj])
                        pyl.imshow(Cut)
                    except:
                        pyl.imshow(cut)

    w = np.where(pred == -1)
    #print(len(w[0]))

    nsp = int(len(w[0])**0.5)
    if len(w[0])>nsp*nsp:
        nsp += 1

    fig = pyl.figure('bad'+appended,figsize=(15,15))
    fig.subplots_adjust(hspace=0,wspace=0)
    gs = gridspec.GridSpec(nsp,nsp)
    for ii in range(nsp):
        for jj in range(nsp):
            if ii*nsp+jj<len(w[0]):
                x,y = int(X[w[0]][ii*nsp+jj])-1,int(Y[w[0]][ii*nsp+jj])-1
                cut = data[y-c:y+c+1,x-c:x+c+1]
                if np.min(cut.shape)<3:
                    continue
                if cut.shape[0]!=2*c+1 or cut.shape[1]!=2*c+1:
                    cut = cutter(x,y,cut,A,B,c,returnCuts)

                if returnCuts:
                    cuts.append(cut)
                    snrs.append(-snr[w[0]][ii*nsp+jj]) #negative for bad sources
                else:
                    try:
                        (z1,z2) = numdisplay.zscale.zscale(cut,contrast = 0.5)
                        normer = interval.ManualInterval(z1,z2)
                        Cut = normer(cut)
                        sp = pyl.subplot(gs[ii,jj])
                        pyl.imshow(Cut)
                    except:
                        pyl.imshow(cut)

    if returnCuts:
        return (cuts,snrs)

    #pyl.show()

def sourceTrim(fn,snr_lim,doBright = True,doBinary = False, aperFWHMmulti = 1.0, medMagDiff = None):

    split = fn.split('-')[-1].split('.fits')[0]
    chip = int(float(split[:3]))

    if doBright :
        o = fn.replace('.fits','_bright_planted.coords')
        file = fn.replace('.fits','_bright_planted.fits')
    elif doBinary :
        o = fn.replace('.fits','_binary_planted.coords')
        file = fn.replace('.fits','_binary_planted.fits')
    else:
        o = fn.replace('.fits','_faint_planted.coords')
        file = fn.replace('.fits','_faint_planted.fits')

    with open(o) as han:
        d = han.readlines()
    if doBinary:
        planted = []
        for i in range(len(d)):
            s = d[i].split()
            (x,y,z,X,Y,Z) = (float(s[0]),float(s[1]),float(s[2]),float(s[3]),float(s[4]),float(s[5]))
            xg = (x*z+X*Z)/(z+Z)
            yg = (y*z+Y*Z)/(z+Z)
            planted.append([xg,yg,z+Z])
        planted = np.array(planted)
    else:
        planted = []
        for i in range(len(d)):
            s = d[i].split()
            planted.append([float(s[0]),float(s[1]),float(s[2])])
        planted = np.array(planted)

    mask_file = '/home/fraserw/idl_progs/hscp9/sextract/mask'+str(chip).zfill(3)+'.fits'


    runSex(file,file,chip,mask_file,svsPath = '',showProgress = False, verbose = False, includeImageMask = True,runSextractor=True)
    pcatalog = scamp.getCatalog(file.replace('.fits','.cat'),paramFile='sextract.param')

    with fits.open(file) as han:
        header = han[0].header
        data = han[1].data


    try:
        FWHM = header['FWHMplant']
    except:
        FWHM = header['fwhmRobust']
    apNum = apertures[round(FWHM*aperFWHMmulti)]


    w = np.where((pcatalog['FLUX_APER(5)'][:,0]>0) & (pcatalog['FLUX_APER(5)'][:,1]>0) &\
                  (pcatalog['FLUX_APER(5)'][:,2]>0) & (pcatalog['FLUX_APER(5)'][:,3]>0) &\
                  (pcatalog['FLUX_APER(5)'][:,4]>0) & (pcatalog['FLUX_AUTO']>0))
    for key in pcatalog:
        pcatalog[key] = pcatalog[key][w]

    psnr = pcatalog['FLUX_APER(5)'][:,apNum]/pcatalog['FLUXERR_APER(5)'][:,apNum]
    w = np.where((pcatalog['X_IMAGE']>50) & (pcatalog['X_IMAGE']<1995) & (pcatalog['Y_IMAGE']>50) & (pcatalog['Y_IMAGE']<4123) \
                 & (psnr>5) & (pcatalog['FWHM_IMAGE']>1.5))
    w1 = np.where((psnr[w]>=snr_lim[0])&(psnr[w]<snr_lim[1]))

    if doBinary:
        dist_max = 2.5
    else:
        dist_max = 1.5

    p = []
    planted_x,planted_y = [],[]
    for i in range(len(planted)):
        dist = ((planted[i,0]-pcatalog['X_IMAGE'][w][w1]+1)**2 + (planted[i,1]-pcatalog['Y_IMAGE'][w][w1]+1)**2)**0.5
        if np.min(dist)<dist_max:
            arg = np.argmin(dist)
            p.append(arg)
            planted_x.append(pcatalog['X_IMAGE'][w][w1][arg])
            planted_y.append(pcatalog['Y_IMAGE'][w][w1][arg])
        #else:
        #    print(i,planted[i],np.min(dist))
    p = np.array(p)
    planted_x = np.array(planted_x)
    planted_y = np.array(planted_y)
    print('Number of planted sources we are making use of:',len(p))

    pmag_aper = -2.5*np.log10(pcatalog['FLUX_APER(5)'][:,apNum][w][w1][p]/header['EXPTIME'])+header['MAGZERO']
    pmag_auto = -2.5*np.log10(pcatalog['FLUX_AUTO'][w][w1][p]/header['EXPTIME'])+header['MAGZERO']
    pmag_diff = pmag_auto - pmag_aper

    if doBright:
        bgf = bgFinder.bgFinder(pmag_diff)
        pmed_mag_diff = bgf.fraserMode(0.4)
        #pmed_mag_diff = getMeanMagDiff(pmag_aper,pmag_diff)
        #if np.isnan(pmed_mag_diff):
        #    pmed_mag_diff = getMeanMagDiff(pmag_aper,pmag_diff,returnMax = True)
    else:
        #use an input one from the planted single sources
        pmed_mag_diff = medMagDiff


    if not np.isnan(pmed_mag_diff):
        pmag_diff = pmag_auto-pmag_aper-pmed_mag_diff

    if doBright:
        plantedMedMagDiff = pmed_mag_diff





    catalog = scamp.getCatalog(fn.replace('.fits','.cat'),paramFile='sextract.param')

    with fits.open(fn) as han:
        header = han[0].header
        data = han[1].data
    (A,B) = data.shape



    w = np.where((catalog['FLUX_APER(5)'][:,0]>0) & (catalog['FLUX_APER(5)'][:,1]>0) &\
                  (catalog['FLUX_APER(5)'][:,2]>0) & (catalog['FLUX_APER(5)'][:,3]>0) &\
                  (catalog['FLUX_APER(5)'][:,4]>0) & (catalog['FLUX_AUTO']>0)&\
                  (catalog['X_IMAGE']>FWHM) & (catalog['X_IMAGE']<B-FWHM) & (catalog['Y_IMAGE']>FWHM) & (catalog['Y_IMAGE']<A-FWHM) )
    for key in catalog:
        catalog[key] = catalog[key][w]

    snr = catalog['FLUX_APER(5)'][:,apNum]/catalog['FLUXERR_APER(5)'][:,apNum]
    w = np.where((snr>5) & (catalog['FWHM_IMAGE']>1.5))
    w1 = np.where((snr[w]>snr_lim[0])&(snr[w]<snr_lim[1]))


    mag_aper = -2.5*np.log10(catalog['FLUX_APER(5)'][:,apNum][w][w1]/header['EXPTIME'])+header['MAGZERO']
    mag_auto = -2.5*np.log10(catalog['FLUX_AUTO'][w][w1]/header['EXPTIME'])+header['MAGZERO']

    if doBright:
        pyl.scatter(mag_auto,mag_auto-mag_aper)
        pyl.scatter(pmag_auto,pmag_auto-pmag_aper)
        pyl.show()
        exit()
    mag_diff = mag_auto-mag_aper-pmed_mag_diff

    if doBright:
        pred = np.ones(len(mag_diff))
        pred[np.where((mag_diff<snr_lim[2])| (mag_diff>np.max(mag_diff)))] = -1

    else:
        args = np.argsort(pmag_auto)
        x = pmag_auto[args]
        y = pmag_diff[args]


        bins = np.zeros(len(x))+abs(snr_lim[2])
        old_i = 0
        for i in range(len(x)):
            if bins[i]<y[i]:
                bins[old_i:] = y[i]
                old_i = i

        ww = np.where(bins==np.max(bins))



        #generate a quadratic function that encompasses the true sources
        b = (bins[ww[0][0]]+0.01+snr_lim_f[2])*((x[ww[0][0]]-np.min(x))**-2)
        X = np.linspace(np.min(mag_auto)-10,np.max(mag_auto)+10,200)
        Y = -snr_lim[2] + b*(X-np.min(x))**2
        ww = np.where(X<np.min(x))
        Y[ww] = -snr_lim[2]
        #now setup the lower limit function with the cutout to include binaries
        Yn=-Y
        ww = np.where(Yn>-0.5)
        Yn[ww]=-0.5


        f = interp.interp1d(X,Y)
        fneg = interp.interp1d(X,Yn)

        pred = -np.ones(len(mag_diff))
        for i in range(len(pred)):
            if mag_diff[i]<f(mag_auto[i]) and mag_diff[i]>(fneg(mag_auto[i])):
                pred[i]=1

    for k in catalog:
        catalog[k] = catalog[k][w][w1]

    if doBright:
        return (catalog,data,pred,mag_auto,mag_diff,pmag_auto,pmag_diff,p,apNum,plantedMedMagDiff)
    elif doBinary:
        return (catalog,data,pred,mag_auto,mag_diff,pmag_auto,pmag_diff,p,apNum)
    else:
        return (catalog,data,pred,mag_auto,mag_diff,pmag_auto,pmag_diff,p,f,fneg,apNum)






snr_lim_b = [60.0,50000.0,-0.5]
snr_lim_bin = [5.0,50000.0,-0.5]
snr_lim_f = [5.0,60.0,-0.2]
aperFWHMmulti = 1.0

doSingle = False
if doSingle:
    fn = 'test_image_data/CORR-0132546-034.fits'


    #planter(fn)
    (catalog_f,data,pred_f,mag_auto,mag_diff,pmag_auto,pmag_diff,p,f,fneg,apNum,pMedMagDiff) = sourceTrim(fn,snr_lim = snr_lim_f,doBright = False,aperFWHMmulti = aperFWHMmulti)
    (catalog_b,data,pred_b,mag_auto,mag_diff,pmag_auto,pmag_diff,p,apNum) = sourceTrim(fn,snr_lim = snr_lim_b,doBright = True,aperFWHMmulti = aperFWHMmulti, medMagDiff = pMedMagDiff)
    (catalog_bin,data,pred_bin,mag_auto_bin,mag_diff_bin,pmag_auto_bin,pmag_diff_bin,p_bin,apNum) = sourceTrim(fn,snr_lim = snr_lim_bin,doBright = False, doBinary=True,aperFWHMmulti = aperFWHMmulti, medMagDiff = pMedMagDiff)

    print('Accepting',len(np.where(pred_b==1)[0])+len(np.where(pred_f==1)[0]),'of',len(pred_b)+len(pred_f),'sources.')


    #output a training set for point_source_classifier.py


    with fits.open(fn) as han:
        data = han[1].data
        header = han[0].header

    (cuts_b,snrs_b) = showSources(pred_b,catalog_b,data,returnCuts = header['BGMEAN'], snr=catalog_b['FLUX_APER(5)'][:,apNum]/catalog_b['FLUXERR_APER(5)'][:,apNum])
    (cuts_f,snrs_f) = showSources(pred_f,catalog_f,data,returnCuts = header['BGMEAN'], snr=catalog_f['FLUX_APER(5)'][:,apNum]/catalog_f['FLUXERR_APER(5)'][:,apNum])

    cuts = []
    for i in range(len(cuts_b)):
        cuts.append(cuts_b[i])
    for i in range(len(cuts_f)):
        cuts.append(cuts_f[i])
    snrs = np.concatenate([snrs_b,snrs_f])

    for i in range(len(cuts)):
        (a,b) = cuts[i].shape
        cuts[i] = cuts[i].reshape(a*b)
    flat = np.array(cuts)


    HDU = fits.PrimaryHDU(flat,header=header)
    SHDU = fits.ImageHDU(snrs)
    List = fits.HDUList([HDU,SHDU])
    List.writeto(fn.replace('.fits','_trainingset.fits'), overwrite=True)
    # finish output of training image


    showSources(pred_b,catalog_b,data,returnCuts = False)
    pyl.show()
    showSources(pred_f,catalog_f,data,returnCuts = False)

    fig = pyl.figure('Scatter')
    w = np.where(pred_f == 1)
    pyl.scatter(mag_auto[w],mag_diff[w],color='r',alpha=0.5,label='good')
    w = np.where(pred_f == -1)
    pyl.scatter(mag_auto[w],mag_diff[w],color='m',alpha=0.5,label='rejected')
    pyl.scatter(pmag_auto,pmag_diff,alpha=0.5,label='planted')
    pyl.scatter(pmag_auto_bin[p_bin],pmag_diff_bin[p_bin],color='y',alpha=0.5,label='binary')

    x = np.linspace(np.min(mag_auto),np.max(mag_auto),100)
    pyl.plot(x,f(x),'y')
    pyl.plot(x,fneg(x))
    pyl.legend(loc=4)
    pyl.show()
else:
    files = ['/media/fraserw/Hammer/DEC2018/02533/HSC-R2/corr/CORR-0155456-034.fits',
             '/media/fraserw/Hammer/DEC2018/02533/HSC-R2/corr/CORR-0155456-056.fits',
             '/media/fraserw/Hammer/DEC2018/02533/HSC-R2/corr/CORR-0155456-058.fits',
             '/media/fraserw/Hammer/DEC2018/02533/HSC-R2/corr/CORR-0155456-072.fits',
             '/media/fraserw/Hammer/DEC2018/02533/HSC-R2/corr/CORR-0155456-083.fits'
            ]
    for ff in range(0,1):#len(files)):

        savesPath = planter(files[ff], saveToRocket=True)
        #savesPath = '/media/fraserw/rocketdata/scratch/psfStars_scratch/'
        fn = savesPath + files[ff].split('/')[-1]
        catPath = files[ff].split('CORR')[0]+'sexSaves/'+files[ff].split('/')[-1].replace('.fits','.cat')
        os.system('cp {} {}'.format(catPath,savesPath))
        os.system('cp {} {}'.format(files[ff],savesPath))

        (catalog_b,data,pred_b,mag_auto_b,mag_diff_b,pmag_auto,pmag_diff,p,apNum,pMedMagDiff) = sourceTrim(fn,snr_lim = snr_lim_b,doBright = True,aperFWHMmulti = aperFWHMmulti)
        (catalog_f,data,pred_f,mag_auto,mag_diff,pmag_auto,pmag_diff,p,f,fneg,apNum) = sourceTrim(fn,snr_lim = snr_lim_f,doBright = False,aperFWHMmulti = aperFWHMmulti, medMagDiff = pMedMagDiff)


        #showSources(pred_b,catalog_b,data,returnCuts = False,appended = '_bright')
        #showSources(pred_f,catalog_f,data,returnCuts = False)

        fig = pyl.figure('Scatter')
        w = np.where(pred_b == 1)
        pyl.scatter(mag_auto_b[w],mag_diff_b[w],color='b',alpha=0.5,label='good')
        w = np.where(pred_b == -1)
        pyl.scatter(mag_auto_b[w],mag_diff_b[w],color='g',alpha=0.5,label='rejected')
        w = np.where(pred_f == 1)
        pyl.scatter(mag_auto[w],mag_diff[w],color='r',alpha=0.5,label='good')
        w = np.where(pred_f == -1)
        pyl.scatter(mag_auto[w],mag_diff[w],color='m',alpha=0.5,label='rejected')

        x = np.linspace(17,25,100)
        pyl.plot(x,f(x))
        pyl.plot(x,fneg(x))
    pyl.show()

    exit()



#thursday at 5:15, 1015 cook st just west of rockland off white stucco house, Dr. Johnston in suite 5
