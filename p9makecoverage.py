import numpy as np
import scipy as sci
from scipy import interpolate as interp
from astropy import wcs
from astropy.io import fits

from paths import *
from p9makebricks import getBrickNum


def brickScale(bns):
    bnu = np.unique(bns)
    out = bns*0.0

    c = 1.0
    for w in bnu:
        out[np.where(bns==w)]=c
        c+=1
    return out

def getCoverBNBXBY(fn,showDiagnostic = False):
    #print(fn)
    with fits.open(fn) as han:
        data = han[1].data
        header = han[1].header

    WCS = wcs.WCS(header)

    (A,B) = data.shape
    x = np.linspace(0.0,B,50.0)
    y = np.linspace(0.0,A,100.0)
    (x_mesh,y_mesh) = np.meshgrid(x,y)


    #the coordinates used in p9extract are numpy/astrpy coordinates
    #and so we use reference 0 not 1 below

    (ra,dec)=WCS.all_pix2world(x_mesh,y_mesh,0)

    f_ra = interp.interp2d(x,y,ra)
    f_dec = interp.interp2d(x,y,dec)

    x = np.arange(0,B,2).astype('float64')
    y = np.arange(0,A,2).astype('float64')


    (x_mesh,y_mesh) = np.meshgrid(x,y)
    (ra_mesh,dec_mesh) = WCS.all_pix2world(x_mesh,y_mesh,0)


    """
    #for interpolation performance metrics
    dra = (f_ra(x,y) - ra_mesh)[0])*3600
    ddec = (f_dec(x,y) - dec_mesh)*3600
    print(np.max(np.abs(dra)),np.max(np.abs(ddec)))
    """

    if not showDiagnostic:
        return getBrickNum(ra_mesh,dec_mesh)

    (bn,bx,by) = getBrickNum(ra_mesh,dec_mesh)
    fig  = pyl.figure()
    fig.add_subplot(221,aspect='equal')
    pyl.imshow(brickScale(bn))
    pyl.title('BN')
    fig.add_subplot(222,aspect='equal')
    pyl.imshow(bx)
    pyl.title('BX')
    fig.add_subplot(223,aspect='equal')
    pyl.imshow(by)
    pyl.title('BY')
    pyl.show()

    return (bn,bx,by)

def getCoverage(exposure, exp_files, coversPath, showCoveragePlot = False):

    coveredBricks = {}
    for ii in range(0,len(exp_files),1):
        print(exp_files[ii],)
        string_chip = exp_files[ii].split('-')[2].split('.')[0]
        chip = int(float(string_chip))

        (brickNums,bxs,bys) = getCoverBNBXBY(exp_files[ii],False)


        if string_chip in maskChips:
            with fits.open('/home/fraserw/idl_progs/hscp9/sextract/mask'+string_chip+'.fits') as han:
                mask_data = han[0].data[::2,::2]
            mask_w = np.where(mask_data == 0)
            bxs[mask_w] = -1
            bys[mask_w] = -1


        brickNums_unique = np.unique(brickNums)
        print(brickNums_unique)
        for bnu in brickNums_unique:
            if bnu not in coveredBricks:
                coveredBricks[bnu] = np.zeros((3600,3600))
            """
            #slow but works
            for k in range(len(bxs)):
                for l in range(len(bxs[k])):
                    if brickNums[k,l] == bnu:
                        coveredBricks[bnu][bxs[k,l],bys[k,l]] = chip+1
            """
            """
            #fast but experimental
            #seems to produce exactly the same thing as above
            for k in range(len(bxs)):
                w = np.where(brickNums[k] == bnu)
                coveredBricks[bnu][bxs[k][w],bys[k][w]] = chip+1000
            """
            #even more experimental
            w = np.where(brickNums == bnu)
            W = np.where(bxs[w] != -1)
            coveredBricks[bnu][bxs[w][W],bys[w][W]] = chip+1000

    for bnu in coveredBricks:
        if not path.exists(coversPath+'/'+str(bnu)):
            os.mkdir(coversPath+'/'+str(bnu))
        np.save(coversPath+'/'+str(bnu)+'/'+exposure+'.cover',coveredBricks[bnu])

    if showCoveragePlot:
        fig = pyl.figure()
        c = 1
        for bnu in coveredBricks:
            fig.add_subplot(3,3,c)
            pyl.imshow(coveredBricks[bnu],interpolation='nearest')
            pyl.title(bnu)
            c+=1
        pyl.show()
    #exit()

if __name__ == "__main__":
    import glob,os
    from os import path
    import pylab as pyl

    coversPath = masterDir+'/coverMaps/'

    if not path.exists(coversPath):
        os.mkdir(coversPath)

    files = glob.glob(masterDir+'/*/HSC-R2/corr/*fits')
    files.sort()


    exposures = []
    for i in files:
        exposures.append(i.split('-')[1])
    exposures = np.array(exposures)
    unique_exposures = np.unique(exposures)

    for exposure in unique_exposures:
        exp_files = []
        for i in range(len(files)):
            if exposure in files[i]:
                exp_files.append(files[i])

        getCoverage(exposure,exp_files,coversPath)
