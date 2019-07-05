from astropy import wcs
from astropy.io import fits
import glob
from paths import *
import numpy as np

masterDir = '/media/fraserw/Hammer/DEC2018'
files = glob.glob(masterDir+'/*/HSC-R2/corr/CORR*fits')
masterDir = '/media/fraserw/Thumber/DEC2018_also'
files += glob.glob(masterDir+'/*/HSC-R2/corr/CORR*fits')
files.sort()

entry = []
for i,fn in enumerate(files):
    with fits.open(fn) as han:
        data = han[0].data
        header0 = han[0].header
        header = han[1].header

    WCS = wcs.WCS(header)
    coords = np.array([[3,3],
              [3,4173],
              [2045,3],
              [2045,4173]])
    c = WCS.all_pix2world(coords,0)
    e = [fn,header0['MJD']]
    for j in range(len(c)):
        e.append(c[j][0])
        e.append(c[j][1])
    entry.append(e)
    print(i+1,len(files),fn)

with open(masterDir+'/corners.txt','w+') as han:
    for i in range(len(entry)):
        han.write("{} {}".format(entry[i][0],str(entry[i][1])))
        for j in range(2,len(entry[i]),2):
            han.write(" {:.7f} {:.7f}".format(entry[i][j],entry[i][j+1]))
        han.write('\n')

#python blinky.py 43090
