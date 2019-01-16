
import scipy as sci, numpy as np, pylab as pyl, glob, os
from os import path
from paths import *
from astropy.io import fits
from astropy.visualization import interval
from astropy import wcs
from stsci import numdisplay
import multiprocessing as multip
import time
import scamp
import pickle

from catObj import catObj
apertures = {2:0,3:0,4:0,5:0,6:1,7:1,8:2,9:2,10:3,11:3,12:4,13:4}


def runSex(file,fn,chip,mask_file,showProgress = False, verbose =  False, includeImageMask = False):

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


            w = np.where(han2>=4000)
            try:
                mask_data[w] = 0.0

            except:
                print(np.max(w[0]),np.max(w[1]))
                exit()

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


    if not path.isfile('default.conv'):
        scamp.makeParFiles.writeConv(overwrite=True)
    if not path.isfile('subaru_LDAC.sex'):
        os.system('cp /home/fraserw/idl_progs/hscp9/sextract/subaru_LDAC.sex .')
    if not path.isfile('sextract.param'):
        os.system('cp /home/fraserw/idl_progs/hscp9/sextract/sextract.param .')


    if showProgress:
        pyl.imshow(data, interpolation='nearest', cmap='gray', origin='lower')
        pyl.colorbar()
        pyl.show()

    if mask is not None:
        scamp.runSex('subaru_LDAC.sex', file+'[0]' ,options={'CATALOG_NAME':savesPath+fn.replace('.fits','.cat'),'WEIGHT_IMAGE':mask,'WEIGHT_TYPE':'map_weight'},verbose=verbose)
    else:
        scamp.runSex('subaru_LDAC.sex', file+'[0]' ,options={'CATALOG_NAME':savesPath+fn.replace('.fits','.cat')},verbose=verbose)

    catalog = scamp.getCatalog(savesPath+fn.replace('.fits','.cat'),paramFile='sextract.param')

    if catObject.seeing < 0:
        #need to estimate seeing because header value is non-sense
        FWHM_IMAGE = np.sort(catalog['FWHM_IMAGE'][np.where((catalog['X_IMAGE']>3) & (catalog['X_IMAGE']<2045) & (catalog['Y_IMAGE']>3) & (catalog['Y_IMAGE']<2045)& (catalog['FLUX_APER(5)'][:,apNum]/catalog['FLUXERR_APER(5)'][:,apNum]>3) )])
        #fwhm_mode = FWHM_IMAGE[len(FWHM_IMAGE)/2]
        fwhm_median = np.median(FWHM_IMAGE)
        catObject.seeing = fwhm_median
        apNum = apertures[round(catObject.seeing)]


    #get rid of the flux=0 sources
    w = np.where(catalog['FLUX_APER(5)'][:,apNum]>0)
    for key in catalog:
        catalog[key] = catalog[key][w]

    snr = catalog['FLUX_APER(5)'][:,apNum]/catalog['FLUXERR_APER(5)'][:,apNum]

    if catObject.seeing>0:
        w = np.where((catalog['X_IMAGE']>3) & (catalog['X_IMAGE']<2045) & (catalog['Y_IMAGE']>3) & (catalog['Y_IMAGE']<2045) & (catalog['FWHM_IMAGE']<catObject.seeing*10.0) & (snr>3) )
    else:
        w = np.where((catalog['X_IMAGE']>3) & (catalog['X_IMAGE']<2045) & (catalog['Y_IMAGE']>3) & (catalog['Y_IMAGE']<2045) & (catalog['FWHM_IMAGE']<4.0) & (snr>3) )


    catObject.fwhm_image = catalog['FWHM_IMAGE'][w]
    catObject.x = catalog['X_IMAGE'][w]-1.0
    catObject.y = catalog['Y_IMAGE'][w]-1.0
    catObject.flux = catalog['FLUX_APER(5)'][:,apNum][w]
    catObject.snr = snr[w]
    catObject.jd = 2400000.5+header['MJD']+header['EXPTIME']/(24.0*3600.0)


    WCS = wcs.WCS(header1)
    (ra,dec) = WCS.all_pix2world(catObject.x,catObject.y,0)
    catObject.ra = ra
    catObject.dec = dec
    catObject.mag = 2.5*np.log10(catObject.flux/header['EXPTIME'])+header['MAGZERO']

    pickle.dump(catObject, open(savesPath+fn.replace('.fits','.sex_save'),'wb'))

    if includeImageMask:
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


#sep.set_extract_pixstack(sepPixStack)


savesPath = sourceDir+'/sexSaves/'

files = glob.glob(sourceDir+'/*fits')
files.sort()

"""
Files = []
for i in range(len(files)-1,-1,-1):
    Files.append(files[i])
files = Files[:]
"""

if not path.exists(savesPath):
    os.mkdir(savesPath)

mask_files = glob.glob(masksDir+'/mask*fits')
mask_files.sort()

showProgress = False




singleThread = False
if singleThread:
    for i in range(len(files)-1,-1,-1):
        fn = files[i].split('/')[-1]
        chip = int(float( fn.split('-')[2].split('.')[0]))
        file = files[i]

        print(file)
        mask_file = '/home/fraserw/idl_progs/hscp9/sextract/mask'+str(chip).zfill(3)+'.fits'

        #print(fn,chip)
        catalog = runSex(file,fn,chip,mask_file,showProgress = False, verbose = False, includeImageMask = True)

    exit()

else:
    numCores = 12


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



            processes.append(multip.Process(target=runSex,args=(file,fn,chip,mask_file,False,False,True)))
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
