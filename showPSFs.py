from astropy.io import fits
import glob
import numpy as np, pylab as pyl
from astropy.visualization import interval
from trippy import tzscale


def click(event):
    global sps, bads

    for i in range(len(sps)):
        for j in range(len(sps[i])):
            if event.inaxes == sps[i][j]:
                bads[i][j] = bads[i][j]!=True
                if bads[i][j]:
                    for spine in sps[i][j].spines.values():
                        spine.set_edgecolor('red')
                else:
                    for spine in sps[i][j].spines.values():
                        spine.set_edgecolor('black')
    pyl.draw()

chips = ['034','056','058','072','083']

files = glob.glob('/media/fraserw/rocketdata/DEC2018/psfStars/*.psf.fits')
files.sort()
#files = files[-27:]

unique_ims = []
for i in files:
    n = i.split('-')[1]
    if n not in unique_ims:
        unique_ims.append(n)
prefix = i.split('-')[0]
postfix='.psf.fits'

w = 19
h = 5


notes = []

n = 0
while n < len(unique_ims):
    fig = pyl.figure('Lookup Tables',figsize=(32,12))
    fig.subplots_adjust(hspace=0, wspace=0)
    cid = fig.canvas.mpl_connect('button_press_event', click)

    sps = []
    bads = []
    names = []
    for i in range(h):
        sps.append([])
        bads.append([])
        names.append([])
        for j in range(w):
            sps[i].append(pyl.subplot2grid((h, w), (i, j), xticklabels = '', yticklabels=''))
            bads[i].append(False)
            names[i].append(None)

    #for i in range(h):
    #    for j in range(w):
    #        bads[i][j] = False

    for i in range(h):
        for j in range(w):
            if n+j==len(unique_ims):
                break

            im = prefix+'-'+unique_ims[n+j]+'-'+chips[i]+postfix
            names[i][j] = im

            try:
                with fits.open(im) as han:
                    lu = han[0].data

                    (z1,z2) = tzscale.zscale(lu,contrast = 0.5)
                    normer = interval.ManualInterval(z1,z2)
                    d = normer(lu)
                    sps[i][j].imshow(d,interpolation = 'nearest')
            except:
                sps[i][j].imshow(np.zeros((30,30)),interpolation = 'nearest')

    n+=w
    if n==len(unique_ims):
        n = len(unique_ims)
        break
    print(n,len(unique_ims))
    pyl.show()

    for i in range(len(bads)):
        for j in range(len(bads[i])):
            if bads[i][j]:
                print(names[i][j])
                notes.append(names[i][j])
print("Redo these ones:")
for i in range(len(notes)):
    print(notes[i])
