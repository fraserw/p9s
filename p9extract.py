import scipy as sci, numpy as np, pylab as pyl, sep, glob, os
from os import path
from paths import *
from astropy.io import fits
from astropy.visualization import interval
from stsci import numdisplay
import multiprocessing as multip
import time

def runSex(file,fn,chip,mask_file,showProgress = False):

    if path.isfile(mask_file):
        with fits.open(mask_file) as han:
            mask = han[0].data==0
    else:
        mask = None

    with fits.open(files[i]) as han:
        data = han[1].data.astype('float64')
        header = han[0].header

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

    t1 = time.time()
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

    t2 = time.time()

    np.save(savesPath+fn.replace('.fits','.sex_save'), obj)

    print(chip,t2-t1,len(obj),bg.globalrms)
    print



sep.set_extract_pixstack(sepPixStack)


savesPath = sourceDir+'/sexSaves/'

files = glob.glob(sourceDir+'/*fits')
files.sort()

if not path.exists(savesPath):
    os.mkdir(savesPath)

mask_files = glob.glob(masksDir+'/mask*fits')
mask_files.sort()

showProgress = False



singleChipTest = False
if singleChipTest:

    i=42
    fn = files[i].split('/')[-1]
    chip = int(float( fn.split('-')[2].split('.')[0]))
    file = files[i]
    mask_file = '/home/fraserw/idl_progs/hscp9/sextract/mask'+str(chip).zfill(3)+'.fits'

    print(fn,chip)
    runSex(file,fn,chip,mask_file,showProgress = False)
    exit()

numCores = 2


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
