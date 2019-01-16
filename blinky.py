#! /usr/bin/env/python

from paths import *
import pylab as pyl
import numpy as np
from stsci import numdisplay
from astropy.io import fits
from astropy.visualization import interval
import glob,pickle
from matplotlib.patches import Circle
from astropy import wcs

def drawReticle(subp,xe,ye):
    subplots[0].plot([xe-20,xe-5],[ye,ye],'r-')
    subplots[0].plot([xe+5,xe+20],[ye,ye],'r-')
    subplots[0].plot([xe,xe],[ye-20,ye-5],'r-')
    subplots[0].plot([xe,xe],[ye+5,ye+20],'r-')

def draw(subplots,normed,counter,xs,ys,X,Y,cut_width):
    subplots[0].cla()
    subplots[0].imshow(normed[counter])
    drawReticle(subplots[0],cut_width+xs[counter]-X[counter][0],cut_width+ys[counter]-Y[counter][0])
    #subplots[0].plot([cut_width-20,cut_width-5],[cut_width,cut_width],'r-')
    #subplots[0].plot([cut_width+5,cut_width+20],[cut_width,cut_width],'r-')
    #subplots[0].plot([cut_width,cut_width],[cut_width-20,cut_width-5],'r-')
    #ubplots[0].plot([cut_width,cut_width],[cut_width+5,cut_width+20],'r-')
    subplots[0].set_ylim(subplots[0].get_ylim()[::-1])

    for i in range(len(X)):
        if i != counter:
            #subplots[0].scatter(X[counter][i]-xs[counter] + cut_width, Y[counter][i]-ys[counter] + cut_width, marker = 'o', edgecolor = 'r', facecolor = 'none')
            subplots[0].scatter((X[0][i]-xs[0]) + cut_width, (Y[0][i]-ys[0]) + cut_width, marker = 'o', edgecolor = 'r', facecolor = 'none')


def blinky(event):
    global subplots,normed, patches, counter,nsp,xs,ys,X,Y

    if event.key in ['a','tab','d']:
        for ii in range(len(patches)):
            subplots[ii+1].patches[-1].set_edgecolor('r')

    if event.key in ['a','tab']:
        counter += 1
        counter = counter%nsp**2
        subplots[counter+1].patches[-1].set_edgecolor('y')
    elif event.key in ['d']:
        counter -= 1
        counter = counter%nsp**2
        subplots[counter+1].patches[-1].set_edgecolor('y')

    if event.key in ['a','tab','d']:
        draw(subplots,normed,counter,xs,ys,X,Y,cut_width)

        pyl.draw()

    if event.key == 'g':
        print('Distant Object!')
    if event.key == 'c':
        print('Bad Object.')

    if event.key in ['g','c']:
        pyl.close()

    if event.key == 'h':
        print('    g - real object')
        print('    c - bad  object')
        print('    a - blink backwards')
        print('d/tab - blink forwards')


cut_width = 300


blinksPath = masterDir+'/blinks'
blinkFiles = glob.glob(blinksPath+'/*')
blinkFiles.sort()

bf = '/media/fraserw/Hammer/DEC2018/blinks/blink.42097'
with open(bf,'rb') as han:
    mover_details = pickle.load(han)

for mi in range(len(mover_details)):

    """
    xs = [1033.45324707,
          1015.01885986,
           923.447021484,
           1085.48388672]
    ys = [1532.95202637,
          1538.46154785,
          1806.71923828,
          1255.5189209]
    ims = ['CORR-0131564-042.fits',
           'CORR-0131930-042.fits',
           'CORR-0132062-042.fits',
           'CORR-0132584-042.fits']
    """
    xs = []
    ys = []
    ims = []
    for i in range(len(mover_details[mi])):
        ims.append(mover_details[mi][i][1].replace('sex_save','fits'))
        xs.append(mover_details[mi][i][2])
        ys.append(mover_details[mi][i][3])



    fig = pyl.figure('Blinky',figsize=(29,14))

    nsp = len(ims)**0.5
    if nsp != int(nsp):
        nsp = int(nsp)+1
    else:
        nsp = int(nsp)

    subplots = []
    subplots.append( pyl.subplot2grid((nsp, nsp+2), (0, 0) ,rowspan = 2, colspan = 2))

    i = 0
    while i<nsp:
        j = 0
        while j<nsp:
            if i*nsp+j<len(ims):
                subplots.append( pyl.subplot2grid((nsp, nsp+2), (i, j+2),aspect='equal') )
            j+=1
        i+=1

    #for i in range(len(subplots)):
    #    subplots[i].invert_yaxis()

    cutouts = []
    normers = []
    normed = []
    wcses = []
    X = []
    Y = []
    datas = []
    headers = []
    for i,im in enumerate(ims):
        print(masterDir+'/*/HSC-R2/corr/'+im,xs[i],ys[i])
        fn = glob.glob(masterDir+'/*/HSC-R2/corr/'+im)[0]
        with fits.open(fn) as han:
            data = han[1].data
            header = han[1].header
        datas.append(data)
        wcses.append(wcs.WCS(header))

    for i,im in enumerate(ims):
        X.append([])
        Y.append([])
        for j in range(len(wcses)):
            if i == j:
                X[-1].append(xs[i])
                Y[-1].append(ys[i])
            else:
                (ra,dec) = wcses[j].all_pix2world(xs[j],ys[j],0)
                (xx,yy) = wcses[i].all_world2pix(ra,dec,0)
                X[-1].append(xx)
                Y[-1].append(yy)

        #print(np.median(datas[i][1850:2300,650:1390]))
        (A,B) = datas[i].shape
        big = np.zeros((A+cut_width*2+1,B+cut_width*2+1)).astype('float64')+np.median(datas[i][1850:2300,650:1390])
        big[cut_width:A+cut_width,cut_width:cut_width+B] = datas[i]
        cut = big[int(Y[i][0]):int(Y[i][0])+2*cut_width,int(X[i][0]):int(X[i][0])+2*cut_width]
        cutouts.append(np.copy(cut))

        (z1,z2) = numdisplay.zscale.zscale(cutouts[-1])
        normers.append( interval.ManualInterval(z1,z2))
        normed.append(normers[-1](cutouts[-1]))


    X = np.array(X)
    Y = np.array(Y)

    #subplots[0].imshow(normed[0])
    #subplots[0].plot([cut_width-20,cut_width-5],[cut_width,cut_width],'r-')
    #subplots[0].plot([cut_width+5,cut_width+20],[cut_width,cut_width],'r-')
    #ubplots[0].plot([cut_width,cut_width],[cut_width-20,cut_width-5],'r-')
    #subplots[0].plot([cut_width,cut_width],[cut_width+5,cut_width+20],'r-')
    #subplots[0].set_ylim(subplots[0].get_ylim()[::-1])

    counter = 0
    draw(subplots,normed,counter,xs,ys,X,Y,cut_width)

    patches = []
    for i in range(len(cutouts)):
        subplots[i+1].imshow(normed[i])
        subplots[i+1].set_ylim(subplots[i+1].get_ylim()[::-1])
        print(cut_width+(xs[i]-X[0][i]),cut_width+(ys[i]-Y[0][i]))
        patches.append(Circle((cut_width+(xs[i]-X[i][0]),cut_width+(ys[i]-Y[i][0])),radius=8, lw = 2,edgecolor='r',facecolor='none'))
        subplots[i+1].add_patch(patches[-1])

    patches[0].set_edgecolor('y')
    cid = fig.canvas.mpl_connect('key_press_event', blinky)

    pyl.show()

    #
    #draw(subplots,normed,cut_width)
