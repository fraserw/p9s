import numpy as np
import pickle
from catObj import *

d2r = np.pi/180.0
def getBrickNum(ra_in,dec_in):
    if type(ra_in) == type(4.0) and type(dec_in) == type(4.0):
        r = (ra_in+360.0)%360.0
        decm = int(dec_in)+.5*dec_in/max(abs(dec_in),0.00001)
        decb = int(dec_in+90.0)
        by = int((int((dec_in-int(dec_in))*3600.)+3600.)%3600.)


        rawidth = 360./int((360.*np.cos(decm*d2r)+.5))
        rab = int(r/rawidth)
        bx = int((r-rab*rawidth)/rawidth*3600.)

        brick = float(rab)*1000.0+decb

        return (int(brick),bx,by)
    else:

        ra=(ra_in+360.0)%360.0
        decm=dec_in.astype('int')+.5*dec_in/np.maximum(np.abs(dec_in),0.00001)
        decb=(dec_in+90.0).astype('int')
        by=((   ( (   dec_in-dec_in.astype('int')   )*3600.).astype('int')+3600.)%3600.).astype('int')

        rawidth=360./(360.*np.cos(decm*d2r)+.5).astype('int')
        rab=(ra/rawidth).astype('int')
        bx=((ra-rab*rawidth)/rawidth*3600.).astype('int')

        brick=rab*1000.0+decb

        return (brick.astype('int'),bx,by)

if __name__ == "__main__":

    testing = False
    if testing:
        ra = np.linspace(0,360,100).astype('float64')
        dec = np.linspace(-90,90,100).astype('float64')
        #print(ra,dec)
        print(getBrickNum(212.2,47.6))
        print(getBrickNum(ra,dec))

        exit()

    import glob,pickle,os,sys
    from os import path
    from paths import *


    if len(sys.argv[1])==1:
        if sys.argv[1] != '--diff':
            print('Pass --diff to open the difference catalog.')
            exit()


        saveFiles = glob.glob(masterDir+'/*/HSC-R2/corr/sexSaves/*sex_save')
        saveFiles.sort()

        bricksPath = masterDir+'/bricks'

        if not path.exists(bricksPath):
            os.mkdir(bricksPath)

        bricks = {}
        for i in range(len(saveFiles)):
            sf = saveFiles[i]
            print(sf,i+1,len(saveFiles))

            image = sf.split('/')[-1].split('.fits')[0]

            catalog = pickle.load(open(sf,'rb'))
            catalog.brick = getBrickNum(catalog.ra,catalog.dec)[0]
            #print(catalog.brick)

            unique_bricks = np.unique(catalog.brick)
            for ubn in unique_bricks:
                if ubn not in bricks:
                    bricks[ubn] = brickObj()

                w = np.where(catalog.brick == ubn)

                bricks[ubn].x = np.concatenate([bricks[ubn].x,catalog.x[w]])
                bricks[ubn].y = np.concatenate([bricks[ubn].y,catalog.y[w]])
                bricks[ubn].ra = np.concatenate([bricks[ubn].ra,catalog.ra[w]])
                bricks[ubn].dec = np.concatenate([bricks[ubn].dec,catalog.dec[w]])
                bricks[ubn].flux = np.concatenate([bricks[ubn].flux,catalog.flux[w]])
                bricks[ubn].snr = np.concatenate([bricks[ubn].snr,catalog.snr[w]])
                bricks[ubn].mag = np.concatenate([bricks[ubn].mag,catalog.mag[w]])
                bricks[ubn].fwhm_image = np.concatenate([bricks[ubn].fwhm_image,catalog.fwhm_image[w]])
                for j in range(len(catalog.x[w])):
                    bricks[ubn].jd.append(catalog.jd)
                    bricks[ubn].seeing.append(catalog.seeing)
                    bricks[ubn].ellip.append(catalog.ellip)
                    bricks[ubn].astrms.append(catalog.astrms)
                    bricks[ubn].images.append(image)

        for bnu in bricks:
            bricks[bnu].jd = np.array(bricks[bnu].jd)
            bricks[bnu].seeing = np.array(bricks[bnu].seeing)
            bricks[bnu].ellip = np.array(bricks[bnu].ellip)
            bricks[bnu].astrms = np.array(bricks[bnu].astrms)
            bricks[bnu].images = np.array(bricks[bnu].images)

            pickle.dump(bricks[bnu],open(bricksPath+'/'+str(bnu)+'.brick','wb'))
    else:
        catalogPath = masterDir+'/DiffCatalog/Candidates_R2_with_frame_xy.dat'
        bricksPath = masterDir+'/DiffCatalog/bricks'
        if not path.exists(bricksPath):
            os.mkdir(bricksPath)

        catalog = catObj(catalogPath)
        catalog.ra*=r2d
        catalog.dec*=r2d
        catalog.brick = getBrickNum(catalog.ra,catalog.dec)[0]
        #print(catalog.brick)
        #exit()

        unique_bricks = np.unique(catalog.brick)


        bricks = {}
        for ubn in unique_bricks:
            if ubn not in bricks:
                bricks[ubn] = brickObj()

            w = np.where(catalog.brick == ubn)

            bricks[ubn].x = np.concatenate([bricks[ubn].x,catalog.x[w]])
            bricks[ubn].y = np.concatenate([bricks[ubn].y,catalog.y[w]])
            bricks[ubn].ra = np.concatenate([bricks[ubn].ra,catalog.ra[w]])
            bricks[ubn].dec = np.concatenate([bricks[ubn].dec,catalog.dec[w]])
            bricks[ubn].jd = np.concatenate([bricks[ubn].jd,catalog.jd[w]])
            bricks[ubn].snr = np.concatenate([bricks[ubn].snr,catalog.snr[w]])
            bricks[ubn].mag = np.concatenate([bricks[ubn].mag,catalog.mag[w]])
            bricks[ubn].fwhm_image = np.concatenate([bricks[ubn].fwhm_image,catalog.fwhm_image[w]])
            bricks[ubn].images = np.concatenate([bricks[ubn].images,catalog.images[w]])

        for ubn in bricks:
            bricks[ubn].flux = np.zeros(len(bricks[ubn].jd))
            bricks[ubn].seeing = np.zeros(len(bricks[ubn].jd))
            bricks[ubn].ellip = np.zeros(len(bricks[ubn].jd))
            bricks[ubn].astrms = np.zeros(len(bricks[ubn].jd))
            bricks[ubn].wf = np.ones(len(bricks[ubn].jd),dtype='bool')
            bricks[ubn].ws = np.ones(len(bricks[ubn].jd),dtype='bool')
            bricks[ubn].wstat = np.ones(len(bricks[ubn].jd),dtype='bool')
        #    bricks[ubn].flux = np.concatenate([bricks[ubn].flux,catalog.flux[w]])
        #    bricks[bnu].jd = np.array(bricks[bnu].jd)
        #    bricks[bnu].seeing = np.array(bricks[bnu].seeing)
        #    bricks[bnu].ellip = np.array(bricks[bnu].ellip)
        #    bricks[bnu].astrms = np.array(bricks[bnu].astrms)
        #    bricks[bnu].images = np.array(bricks[bnu].images)

            pickle.dump(bricks[ubn],open(bricksPath+'/'+str(ubn)+'.brick','wb'))
