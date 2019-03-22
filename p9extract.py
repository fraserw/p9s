
import scipy as sci, numpy as np, pylab as pyl, glob, os
from os import path
from paths import *
from astropy.io import fits
from astropy.visualization import interval
from astropy import wcs
from stsci import numdisplay
import multiprocessing as multip
import time
#import scamp
import pickle
from trippy import scamp,bgFinder

from catObj import catObj
apertures = {2:0,3:0,4:0,5:0,6:1,7:1,8:2,9:2,10:3,11:3,12:4,13:4,14:4,15:4,16:4,17:4,18:4,19:4,20:4}


def getMeanMagDiff(maper,diff,bwidth = 1, returnMax = False):
    """
    returnMax is good when the sources are few, and the SNR is high
    """

    bins = np.arange(int(np.min(maper)),int(np.max(maper))+bwidth,bwidth)
    k = (maper-bins[0]).astype('int')
    meds = []
    stds = []
    for ii in range(len(bins)):
        w = np.where(k==ii)
        if returnMax:
            meds.append(np.nanmax(diff[w]))
            stds.append(np.nanstd(diff[w]))
        elif len(w[0])>=5:
            meds.append(np.nanmedian(diff[w]))
            stds.append(np.nanstd(diff[w]))
        else:
            meds.append(np.nan)
            stds.append(np.nan)
    meds = np.array(meds)
    stds = np.array(stds)

    if returnMax:
        return np.median(meds)

    if np.sum(np.isnan(meds))>len(meds)-3:
        return np.nan

    w = np.where(np.isnan(stds)==0)

    meds = meds[w]
    stds = stds[w]

    bins = bins[w]
    weight = 1.0/stds**2
    mean = np.sum(meds*weight)/np.sum(weight)

    return mean


def runSex(file,fn,chip,mask_file,svsPath,showProgress = False, verbose =  False, includeImageMask = False, kron_cut = -0.5, runSextractor=True):

    if path.isfile(mask_file):
        mask = mask_file+'[0]'

        if includeImageMask:
            with fits.open(mask_file) as han:
                mask_data = han[0].data
            #print('Using image mask')
        else:
            mask_data = None

    else:
        mask = None
        mask_data = None


    with fits.open(file) as han:

        header = han[0].header
        header1 = han[1].header
        if includeImageMask:
            han2 = han[2].data
            if mask_data is None:
                mask_data = np.ones(han2.shape).astype(han2.dtype)


            w = np.where((han2>32)|(han2==12)) #12 is cosmic rays, everything above 32 seems to be saturation
            try:
                mask_data[w] = 0.0

            except:
                print(np.max(w[0]),np.max(w[1]))
                exit()

            if runSextractor:
                fits.writeto(file+'.mask',mask_data,overwrite = True)
            mask = file+'.mask'

        if showProgress:
            data = han[0].data


    seeing = header['SEEING_MODE']

    try:
        apNum = apertures[round(seeing)]
    except:
        #seeing variable is -9999. This apNum variable will be reset based on sextractor output
        apNum = apertures[2]

    catObject = catObj()
    catObject.seeing = header['SEEING_MODE']
    catObject.ellip = header['ELL_MED']
    catObject.astrms = header['WCS_RMS']


    if showProgress:
        pyl.imshow(data, interpolation='nearest', cmap='gray', origin='lower')
        pyl.colorbar()
        pyl.show()

    if runSextractor:
        if mask is not None:
            scamp.runSex('subaru_LDAC.sex', file+'[0]' ,options={'CATALOG_NAME':svsPath+fn.replace('.fits','.cat'),'WEIGHT_IMAGE':mask,'WEIGHT_TYPE':'map_weight'},verbose=verbose)
        else:
            scamp.runSex('subaru_LDAC.sex', file+'[0]' ,options={'CATALOG_NAME':svsPath+fn.replace('.fits','.cat')},verbose=verbose)

    catalog = scamp.getCatalog(svsPath+fn.replace('.fits','.cat'),paramFile='sextract.param')

    #get rid of the flux=0 sources
    w = np.where((catalog['FLUX_APER(9)'][:,0]>0) & (catalog['FLUX_APER(9)'][:,1]>0) & (catalog['FLUX_APER(9)'][:,2]>0) & (catalog['FLUX_APER(9)'][:,3]>0) & (catalog['FLUX_APER(9)'][:,4]>0)\
                 & (catalog['FLUX_APER(9)'][:,5]>0) & (catalog['FLUX_APER(9)'][:,6]>0)\
                 & (catalog['FLUX_AUTO']>0))
    for key in catalog:
        catalog[key] = catalog[key][w]


    if catObject.seeing <= 0:
        #need to estimate seeing because header value is non-sense
        #use all snr>40 sources, and take the median FWHM_IMAGE value
        FWHM_IMAGE = np.sort(catalog['FWHM_IMAGE'][np.where((catalog['X_IMAGE']>50) & (catalog['X_IMAGE']<1995) & (catalog['Y_IMAGE']>50) & (catalog['Y_IMAGE']<4123) & (catalog['FLUX_APER(9)'][:,apNum]/catalog['FLUXERR_APER(9)'][:,apNum]>40) )])
        #fwhm_mode = FWHM_IMAGE[len(FWHM_IMAGE)/2]
        fwhm_median = np.median(FWHM_IMAGE)
        catObject.seeing = fwhm_median
        apNum = apertures[round(catObject.seeing)]



    #setup cut on Kron magnitude, by getting the median difference between kron and aperture magnitude for star-like objects.
    w = np.where((catalog['X_IMAGE']>50) & (catalog['X_IMAGE']<1995) & (catalog['Y_IMAGE']>50) & (catalog['Y_IMAGE']<4123) &  ((catalog['FLUX_APER(9)'][:,apNum]/catalog['FLUXERR_APER(9)'][:,apNum])>40) & (catalog['FWHM_IMAGE']>1.5))

    mag_aper = -2.5*np.log10(catalog['FLUX_APER(9)'][:,apNum][w]/header['EXPTIME'])+header['MAGZERO']
    mag_auto = -2.5*np.log10(catalog['FLUX_AUTO'][w]/header['EXPTIME'])+header['MAGZERO']

    mag_diff = mag_auto - mag_aper

    bgf = bgFinder.bgFinder(mag_diff)
    med_mag_diff = bgf.fraserMode(0.4)
    #med_mag_diff = getMeanMagDiff(mag_aper,mag_diff)
    #if np.isnan(med_mag_diff):
    #    med_mag_diff = getMeanMagDiff(mag_aper,mag_diff,returnMax = True)

    #cut on position, SNR, and FWHM
    snr = catalog['FLUX_APER(9)'][:,apNum]/catalog['FLUXERR_APER(9)'][:,apNum]
    if catObject.seeing>0:
        w = np.where((catalog['X_IMAGE']>3) & (catalog['X_IMAGE']<2045) & (catalog['Y_IMAGE']>3) & (catalog['Y_IMAGE']<4173) & (catalog['FWHM_IMAGE']<catObject.seeing*5.0) & (snr>3) & (catalog['FWHM_IMAGE']>1.5) & (catalog['A_IMAGE']>1.0) & (catalog['B_IMAGE']>1.0) )
    else:
        w = np.where((catalog['X_IMAGE']>3) & (catalog['X_IMAGE']<2045) & (catalog['Y_IMAGE']>3) & (catalog['Y_IMAGE']<4173) & (catalog['FWHM_IMAGE']<10.0) & (snr>3) & (catalog['FWHM_IMAGE']>1.5) & (catalog['A_IMAGE']>1.5) & (catalog['B_IMAGE']>1.0) ) #now measured in arcseconds rather than in units of FWHM



    #now cut on difference between kron and aperture magnitudes, assuming there were enough sources for the cut
    if not np.isnan(med_mag_diff):
        mag_aper = -2.5*np.log10(catalog['FLUX_APER(9)'][:,apNum][w]/header['EXPTIME'])+header['MAGZERO']
        mag_auto = -2.5*np.log10(catalog['FLUX_AUTO'][w]/header['EXPTIME'])+header['MAGZERO']
        mag_diff = mag_auto-mag_aper-med_mag_diff
        w = [w[0][np.where(mag_diff>kron_cut)]]
        #pyl.scatter(mag_aper,mag_diff)
        #print(len(w[0]))
        #pyl.scatter(mag_aper[np.where(np.abs(mag_diff)<kron_cut)],mag_diff[np.where(np.abs(mag_diff)<kron_cut)])
        #pyl.show()
        #exit()


    catObject.fwhm_image = catalog['FWHM_IMAGE'][w]
    catObject.x = catalog['X_IMAGE'][w]-1.0
    catObject.y = catalog['Y_IMAGE'][w]-1.0
    catObject.flux = catalog['FLUX_APER(9)'][:,apNum][w]
    catObject.snr = snr[w]
    catObject.jd = 2400000.5+header['MJD']+header['EXPTIME']/(24.0*3600.0)


    WCS = wcs.WCS(header1)
    (ra,dec) = WCS.all_pix2world(catObject.x,catObject.y,0)
    catObject.ra = ra
    catObject.dec = dec
    catObject.mag = 2.5*np.log10(catObject.flux/header['EXPTIME'])+header['MAGZERO']

    pickle.dump(catObject, open(svsPath+fn.replace('.fits','.sex_save'),'wb'))

    if includeImageMask and runSextractor:
        os.remove(file+'.mask')

    print(file,len(catObject.ra))
    return catObject



def runSex_sep(file,fn,chip,mask_file,showProgress = False):

    if path.isfile(mask_file):
        with fits.open(mask_file) as han:
            mask = han[0].data==0
    else:
        mask = None

    with fits.open(file) as han:
        mask_data = han[2].data.astype('float64')
        header = han[0].header
        header1 = han[1].header


    WCS = wcs.WCS(header1)

    (A,B) = data.shape
    if A==4716:
        bh = 72
        bw = 64

    else:
        bh = 64
        bw = 72


    if showProgress:
        pyl.imshow(data, interpolation='nearest', cmap='gray', origin='lower')
        pyl.colorbar()
        pyl.show()

    #t1 = time.time()
    bg = sep.Background(data, mask=mask,bw = bw, bh=bh)

    bg_im = bg.back()

    if showProgress:
        pyl.imshow(bg_im, interpolation='nearest', cmap='gray', origin='lower')
        pyl.colorbar()
        pyl.show()



    if showProgress:
        data_sub = np.copy(data) - bg_im
        (z1,z2) = numdisplay.zscale.zscale(data_sub)
        normer = interval.ManualInterval(z1,z2)
        pyl.imshow(normer(data_sub), interpolation='nearest', cmap='gray', origin='lower')
        pyl.colorbar()
        pyl.show()


    #print(bg.globalrms)
    obj = sep.extract(np.copy(data) - bg_im, 1.2,minarea = 3, mask=mask, err = bg.globalrms)
    np.save(savesPath+fn.replace('.fits','.sex_save'), obj)

    #t2 = time.time()



    #print(chip,t2-t1,len(obj),bg.globalrms)
    #print

    (ra,dec) = WCS.all_pix2world(obj['x'],obj['y'],0)

    arr = np.zeros(len(obj['x']), dtype = [('x','float64'), ('y','float64'), ('ra','float64'), ('dec','float64'), ('flux','float64'), ('snr','float64'), ('fwhm','float64')])
    arr['x'] = obj['x']
    arr['y'] = obj['y']
    arr['ra'] = ra
    arr['dec'] = dec
    arr['flux'] = obj['flux']
    arr['snr'] = obj['flux']
    #arr[:,1] = obj['y']
    print(arr[0])
    print(arr[0].shape)
    exit()



    exit()
    return obj


if __name__ == "__main__":
    #sep.set_extract_pixstack(sepPixStack)

    kron_cut = -0.5
    runSextractor = True

    savesPath = sourceDir+'/sexSaves/'

    Files = glob.glob(sourceDir+'/*fits')
    Files.sort()


    if not path.exists(savesPath):
        os.mkdir(savesPath)

    overWrite = True
    if not overWrite:
        files = []
        for i in range(len(Files)):
            s = Files[i].split('/')
            fn = '/'.join(s[:6])+'/sexSaves/'+s[6].replace('.fits','.sex_save')

            if not path.isfile(fn):
                files.append(Files[i])
    else:
        files = Files[:]




    mask_files = glob.glob(masksDir+'/mask*fits')
    mask_files.sort()

    showProgress = False


    if not path.isfile('default.conv'):
        scamp.makeParFiles.writeConv(overwrite=True)
    if not path.isfile('subaru_LDAC.sex'):
        os.system('cp /home/fraserw/idl_progs/hscp9/sextract/subaru_LDAC.sex .')
    if not path.isfile('sextract.param'):
        os.system('cp /home/fraserw/idl_progs/hscp9/sextract/sextract.param .')



    singleThread = False
    if singleThread:
        for i in range(len(files)-1,-1,-1):
            fn = files[i].split('/')[-1]
            chip = int(float( fn.split('-')[2].split('.')[0]))
            file = files[i]

            #if file!='/media/fraserw/Thumber/FEB2018/02231/HSC-R2/corr/CORR-0139062-091.fits': continue
            print(file)
            mask_file = '/home/fraserw/idl_progs/hscp9/sextract/mask'+str(chip).zfill(3)+'.fits'

            #print(fn,chip)
            catalog = runSex(file,fn,chip,mask_file,savesPath,showProgress = False, verbose = True, includeImageMask = True,kron_cut = kron_cut,runSextractor = runSextractor)

        exit()

    else:
        numCores = 11


        #attempt at multiprocessing with runSex which seems to fail
        #using Queue
        q = multip.Queue()
        i = 0
        while i < len(files)+1:

            processes = []
            for j in range(numCores):
                if i+j == len(files):
                    break
                fn = files[i+j].split('/')[-1]
                chip = int(float( fn.split('-')[2].split('.')[0]))
                file = files[i+j]
                mask_file = '/home/fraserw/idl_progs/hscp9/sextract/mask'+str(chip).zfill(3)+'.fits'
                #print(file,chip)



                processes.append(multip.Process(target=runSex,args=(file,fn,chip,mask_file,savesPath,False,False,True,kron_cut,runSextractor)))
                processes[j].start()

            for j in range(len(processes)):
                processes[j].join()

            i+=numCores
    """
    numCores = 2


    #attempt at multiprocessing with runSex_sep which seems to fail
    #using Queue
    q = multip.Queue()
    for i in range(0,len(files),numCores):


        processes = []
        for j in range(numCores):
            if i+j+1 == len(files):
                break
            fn = files[i+j].split('/')[-1]
            chip = int(float( fn.split('-')[2].split('.')[0]))
            file = files[i+j]
            mask_file = '/home/fraserw/idl_progs/hscp9/sextract/mask'+str(chip).zfill(3)+'.fits'
            #print(file,chip)



            processes.append(multip.Process(target=runSex,args=(file,fn,chip,mask_file)))
            processes[j].start()

        for j in range(len(processes)):
            processes[j].join()
    """
