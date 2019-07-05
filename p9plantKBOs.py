import numpy as np, pylab as pyl, scipy as sci
from trippy import psf,scamp
from astropy.io import fits
from stsci import numdisplay
from astropy.visualization import interval
from astropy import wcs
import time
from scipy import interpolate as interp
from p9extract import runSex, getMeanMagDiff
import pickle
from paths import *
import sys,glob,os
import ephem as eph

def getPSFchip(chip):
    if chip in   ['66', '34', '65', '33']:
        return '34'
    elif chip in ['67', '59', '51', '43', '35', '81', '74',\
                  '26', '19', '80', '73', '25', '18', '64',\
                  '56', '48', '40', '32']:
        return '56'
    elif chip in ['58', '50', '42', '57', '49', '41']:
        return str(58)
    elif chip in ['79', '72', '55', '47', '39', '24', '17',\
                  '86', '80', '18', '12', '87', '81', '19',\
                  '13', '82', '75', '20', '14', '60', '62',\
                  '44']:
        return str(72)
    else:
        return str(83)


def getKBORaDec(jd,bodies,observer):
    djd = jd - 2415020.0
    ras = []
    decs = []
    sds = []
    eds = []
    observer.date = djd
    for ii in range(len(bodies)):
        bodies[ii].compute(observer)
        ras.append(float(bodies[ii].a_ra))
        decs.append(float(bodies[ii].a_dec))
        sds.append(bodies[ii].sun_distance)
        eds.append(bodies[ii].earth_distance)

    """
    ##HACK for image CORR-0154888-034.psf.fits
    ras[0] = 107.55356888991243*d2r
    decs[0] = 32.52696569510936*d2r
    """

    return(np.array(ras)*r2d,np.array(decs)*r2d,np.array(sds),np.array(eds))


def plantKBOs(fits_fn = '/media/fraserw/Hammer/DEC2018/02531/HSC-R2/corr/CORR-0154888-034.fits',bodies=[],observer=[], kbos = [], dirInd = ''):
    im = fits_fn.split('/')[-1]
    #cat_fn = fits_fn.replace('corr/','corr/sexSaves/').replace('.fits','.cat')
    chip = int(float(im.split('-')[-1].split('.')[0]))
    psfChip = getPSFchip(str(chip))
    schip = str(chip)
    spsfChip = str(psfChip)
    if len(schip)<2:
        schip = '00'+schip
    elif len(schip)<3:
        schip = '0'+schip
    if len(spsfChip)<2:
        spsfChip = '00'+spsfChip
    elif len(spsfChip)<3:
        spsfChip = '0'+spsfChip

    psf_fn = '/media/fraserw/rocketdata/DEC2018/psfStars/{}'.format(im.replace('.fits','.psf.fits').replace('-'+str(schip)+'.','-'+str(spsfChip)+'.'))


    if chip<100:
        x_high = 2045
        y_high = 4173
    else:
        print('Chip>100')
        x_high = 4173
        y_high = 2045


    #catalog = scamp.getCatalog(cat_fn,paramFile='sextract.param')


    with fits.open(fits_fn) as han:
        data = han[1].data
        header = han[0].header
        header1 = han[1].header

    gain = header['MEDGAIN']
    new_data = np.copy(data)+header['BGMEAN']

    (A,B) = data.shape

    WCS = wcs.WCS(header1)
    (mid_ra,mid_dec) = WCS.all_pix2world(A/2.0,B/2.0,0)

    (kbo_ras,kbo_decs,kbo_sds,kbo_eds) = getKBORaDec(header['MJD']+2400000.5+header['EXPTIME']/(3600.0*24.0),bodies,observer)

    goodPSF = psf.modelPSF(restore = psf_fn)
    fwhm = goodPSF.FWHM()

    w = np.where((np.abs(kbo_ras-mid_ra)<0.25) & (np.abs(kbo_decs-mid_dec)<0.25))
    if len(w[0])==0:
        return ([],[],[],[])

    else:
        x,y = [],[]
        for ii in range(len(w[0])):
            try:
                (xx,yy) = WCS.all_world2pix(kbo_ras[w[0][ii]],kbo_decs[w[0][ii]],0)
            except:
                (xx,yy) = -1.0,-1.0
            x.append(xx)
            y.append(yy)
        x = np.array(x)
        y = np.array(y)

        W = np.where((x>0)&(x<x_high)&(y>0)&(y<y_high))
        kbo_ind = w[0][W[0]]
        kbo_x = x[W[0]]
        kbo_y = y[W[0]]
        kbo_m = kbos[:,-2][kbo_ind]+5.0*np.log10(kbo_sds[kbo_ind]*kbo_eds[kbo_ind])
        print(kbo_m)


        pWidth = int( (goodPSF.psf.shape[0]-1)/2 )

        blank = goodPSF.plant(100.5,100.5,1.0,np.zeros((200,200)).astype('float64')+header['BGMEAN'],addNoise=False,useLinePSF=False,returnModel=False,gain = gain, plantBoxWidth=pWidth)
        flux_scale = np.sum(blank) - 200.0*200.0*header['BGMEAN']
        mag_scale = header['MAGZERO']-2.5*np.log10(flux_scale/header['EXPTIME'])



        #for planting  3<snr<200.0
        bg_flux = np.pi*(2*fwhm)**2*header['BGMEAN']

        multis = 10.0**np.linspace(-4,4,5000.0)
        snr_estimate = multis*flux_scale*(multis*flux_scale+bg_flux)**-0.5

        """
        f = interp.interp1d(snr_estimate,multis)
        low_multi = f(3.0)
        high_multi = f(200.0)
        mag_low = header['MAGZERO']-2.5*np.log10(flux_scale*low_multi/header['EXPTIME'])
        """

        kbo_flux = 10.0**(0.4*(header['MAGZERO']-kbo_m))*header['EXPTIME']
        kbo_multi = kbo_flux/flux_scale

        #f = interp.interp1d(multis,snr_estimate)
        #print(f(kbo_multi),fwhm)
        for i in range(len(kbo_x)):
            x = kbo_x[i]
            y = kbo_y[i]
            v = kbo_multi[i]
            print(i,x,y,v)

            if x<pWidth or x>x_high-pWidth or y<pWidth or y>y_high-pWidth:
                print('here')
                new_data = goodPSF.plant(x,y,v,np.copy(new_data),addNoise=True,useLinePSF=False,returnModel=False,gain = gain,plantBoxWidth=pWidth)
            else:
                a = int(y-pWidth)
                b = int(y+pWidth+1)
                c = int(x-pWidth)
                d = int(x+pWidth+1)
                new_data[a:b,c:d] = goodPSF.plant(x-int(x)+pWidth+0.5,y-int(y)+pWidth+0.5,v,np.copy(new_data[a:b,c:d]),addNoise=True,useLinePSF=False,returnModel=False,gain = gain,plantBoxWidth=pWidth)

        new_data -= header['BGMEAN']

        with fits.open(fits_fn) as han:
            for i in range(len(han[1].data)):
                han[1].data[i] = new_data[i]
            han[0].header['FWHMplant'] = fwhm
            han.writeto('/mnt/ramdisk/junk_{}.fits'.format(dirInd), overwrite = True)

        return (kbo_ind,kbo_x,kbo_y,kbo_multi)


if __name__ == "__main__":

    with open('kbos_to_plant.dat') as han:
        data = han.readlines()
    kbos = []
    for i in range(len(data)):
        o = []
        s = data[i].split()
        for j in range(3,len(s)):
            o.append(float(s[j]))
        kbos.append(o)
    kbos = np.array(kbos)

    bodies = []
    for i in range(len(kbos)):
        kbo = eph.EllipticalBody()
        (a,e,inc,a0,a1,a2,djdM,djd,H,m) = kbos[i]
        kbo._a = a
        kbo._e = e
        kbo._inc = inc
        kbo._M = a0
        kbo._Om = a1
        kbo._om = a2
        kbo._epoch_M = djdM
        kbo._epoch = djd
        bodies.append(kbo)
        del kbo

    observer = eph.Observer()
    observer.lon = '204:31:40.1'
    observer.lat = '19:49:34.0'
    observer.elevation = 4212


    mask_files = glob.glob(masksDir+'/mask*fits')
    mask_files.sort()

    dirInd = '02530'
    if dirInd == '02531':
        dirStr = 'rocketdata/DEC2018'
    elif dirInd in ['02532','02533','02534']:
        dirStr = 'Hammer/DEC2018'
    else:
        dirStr = 'Thumber/DEC2018_also'

    os.system('PS1=$')
    os.system('PROMPT_COMMAND=')
    os.system('echo -en "\033]0;{}\a"'.format(dirInd))
    #files = ['/media/fraserw/Hammer/DEC2018/02531/HSC-R2/corr/CORR-0154534-100.fits']
    files = glob.glob('/media/fraserw/{}/{}/HSC-R2/corr/CORR*.fits'.format(dirStr,dirInd))
    files.sort()


    svsPath = '/media/fraserw/rocketdata/DEC2018/sexSaves_plant/'

    g = glob.glob(svsPath+'/*sex_save')
    alreadyDone = []
    for i in g:
        s = i.split('/')[-1].replace('.sex_save','.fits')
        alreadyDone.append(s)
    alreadyDone.sort()


    for ff,fits_fn in enumerate(files):
        chip = fits_fn.split('-')[-1].split('.')[0]
        fn = fits_fn.split('/')[-1]
        print(fn)


        if fn in alreadyDone:
            print('   ...already done.\n')
            continue
        planted = plantKBOs(fits_fn, bodies, observer, kbos, dirInd)

        if len(planted[0])>0:
            with open(svsPath+fn.replace('.fits','.planted'),'w+') as outhan:
                for i in range(len(planted[0])):
                    print(planted[0][i],planted[1][i],planted[2][i],planted[3][i],file=outhan)

            mask_file = '/home/fraserw/idl_progs/hscp9/sextract/mask'+str(chip).zfill(3)+'.fits'
            #mask_file = '/mnt/ramdisk/mask'+str(chip).zfill(3)+'.fits'
            runSex('/mnt/ramdisk/junk_{}.fits'.format(dirInd),fn,int(float(chip)),mask_file,svsPath,showProgress = False, verbose = False, includeImageMask = True, kron_cut = -0.5, runSextractor=True)
        else:
            comm = 'ln -s /media/fraserw/rocketdata/DEC2018/sexSaves/{} /media/fraserw/rocketdata/DEC2018/sexSaves_plant/'.format(fn.replace('.fits','.sex_save'))
            print(comm)
            os.system(comm)
"""
    files = ['/media/fraserw/Hammer/DEC2018/02531/HSC-R2/corr/CORR-0154888-034.fits']

from catObj import *
import pickle
sf = '/media/fraserw/rocketdata/DEC2018/sexSaves_plant/CORR-0154888-034.sex_save'
cat = pickle.load(open(sf,'rb'))
import numpy as np
arg = np.argsort(((cat.x-312.754110761)**2 + (cat.y-2001.7009442)**2)**0.5)
cat.x[arg[0]]
cat.y[arg[0]]
"""

#opening a ram disk: sudo mount -t tmpfs -o rw,size=2G tmpfs /mnt/ramdisk
