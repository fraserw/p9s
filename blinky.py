#! /usr/bin/env/python

from paths import *
import pylab as pyl
import numpy as np
from stsci import numdisplay
from astropy.io import fits
from astropy.visualization import interval
import glob,pickle,os,sys
from matplotlib.patches import Circle
from astropy import wcs
import bgFinder
from os import path
from inRect import isInRect


def findBGImage(x,jd,corners):
    otherIms = []
    for kk in corners:
        if isInRect(x,corners[kk][1]) and abs(corners[kk][0]-jd)>0.5:
            otherIms.append(kk)

    return otherIms


def drawReticle(subp,xe,ye,fmt = 'r:'):
    subplots[0].plot([xe-20,xe-10],[ye,ye],fmt)
    subplots[0].plot([xe+10,xe+20],[ye,ye],fmt)
    subplots[0].plot([xe,xe],[ye-20,ye-10],fmt)
    subplots[0].plot([xe,xe],[ye+10,ye+20],fmt)

def draw(subplots,normed,counter,xs,ys,X,Y,cut_width,fieldCut = None,fmt='r-'):
    subplots[0].cla()
    if fieldCut is None:
        subplots[0].imshow(normed[counter])
    else:
        subplots[0].imshow(fieldCut)
    drawReticle(subplots[0],cut_width+xs[counter]-X[counter][0],cut_width+ys[counter]-Y[counter][0],fmt)
    subplots[0].set_ylim(subplots[0].get_ylim()[::-1])

    for i in range(len(X)):
        if i != counter:
            subplots[0].scatter(X[counter][i]-X[counter][0] + cut_width, Y[counter][i]-Y[counter][0] + cut_width, marker = 'o', edgecolor = 'r', facecolor = 'none', s =600)
            #subplots[0].scatter((X[0][i]-xs[0]) + cut_width, (Y[0][i]-ys[0]) + cut_width, marker = 'o', edgecolor = 'r', facecolor = 'none',s=400)


def blinky(event):
    global subplots,normed, patches, counter,nsp,ims,xs,ys,ras,decs,jds,X,Y,dist_low,dist_high,distantCandidate,corners, cut_width, possCounter, possIms, possDrawing, fieldCuts, contrast

    if event.key == 'q':
        exit()

    if event.key in ['a','tab','d']:
        possIms = []
        possDrawing = False
        possCounter = 0
        for ii in range(len(patches)):
            subplots[ii+1].patches[-1].set_edgecolor('r')

    if event.key in ['a','tab']:
        counter += 1
        counter = counter%len(normed)
        subplots[counter+1].patches[-1].set_edgecolor('y')
    elif event.key in ['d']:
        counter -= 1
        counter = counter%len(normed)
        subplots[counter+1].patches[-1].set_edgecolor('y')

    if event.key in ['a','tab','d']:
        draw(subplots,normed,counter,xs,ys,X,Y,cut_width)
        pyl.draw()

    if event.key == 'g':
        print('Distant Object!',dist_low,dist_high)
        distantCandidate = True
    if event.key == 'c':
        print('Bad Object.')

    if event.key in ['g','c']:
        pyl.close()

    if event.key in ['b']:
        if len(possIms) == 0:
            print("Getting field image of ",ims[counter])
            possImss = findBGImage([ras[counter],decs[counter]],jds[counter],corners)
            possIms = []
            for k in range(len(possImss)):
                if possImss[k] not in ims:
                    possIms.append(possImss[k])
            print('    Have',len(possIms),'images to choose from.\n')
            possDrawing = True

        if possDrawing and len(possIms)>0:

            print('Loading',possIms[possCounter])

            if len(fieldCuts) >= possCounter-1:

                with fits.open(possIms[possCounter]) as han:
                    fieldData = han[1].data
                    fieldHeader = han[1].header
                fWCS = wcs.WCS(fieldHeader)

                (xx,yy) = fWCS.all_world2pix(ras[0],decs[0],0)
                #print(fWCS.all_pix2world(xx,yy,0),possCounter,'&&')
                (A,B) = fieldData.shape

                bg = 0.0
                big = np.zeros((A+cut_width*4+1,B+cut_width*4+1)).astype('float64')+bg
                big[cut_width*2:A+cut_width*2,cut_width*2:cut_width*2+B] = fieldData
                cut = big[int(yy)+cut_width:int(yy)+3*cut_width,int(xx)+cut_width:int(xx)+3*cut_width]


                (z1,z2) = numdisplay.zscale.zscale(fieldData,contrast = contrast)

                normer = interval.ManualInterval(z1,z2)
                fieldCut = normer(cut)
                fieldCut[:5,:5]=1.0
                fieldCut[-5:,:5]=1.0
                fieldCut[-5:,-5:]=1.0
                fieldCut[:5,-5:]=1.0
                fieldCuts.append(np.copy(fieldCut))
            else:
                fieldCut = fieldCuts[possCounter]
            draw(subplots,normed,counter,xs,ys,X,Y,cut_width,fieldCut = fieldCut,fmt='w-')

            possCounter = (possCounter+1)%len(possIms)

            pyl.draw()


        else:
            print("...No background images available!\n")
    else:
        possDrawing = False
        possIms = []
        possCounter = 0

    if event.key == 'h':
        print('    g - real object')
        print('    c - bad  object')
        print('    a - blink backwards')
        print('d/tab - blink forwards')
        print('    q - quit without saving a candidates file.')

widths = [200,300,500,700,900,1000,1200,1400,1600]
contrast = 0.4

doFast = False
doVeryFast = False
if '--fast' in sys.argv:
    doFast = True
    print('Using fast_bricks')
elif '--veryfast' in sys.argv:
    doVeryFast = True
    print('Using veryfast_bricks')

"""
blinksPath = masterDir+'/blinks'
if doFast:
    blinksPath = masterDir+'/fast_blinks'

blinkFiles = glob.glob(blinksPath+'/*')
blinkFiles.sort()
"""

candidatesPath = masterDir+'/candidates'
if doFast:
    candidatesPath = masterDir+'/fast_candidates'
elif doVeryFast:
    candidatesPath = masterDir+'/veryfast_candidates'
if not path.exists(candidatesPath):
    os.mkdir(candidatesPath)


#reseting master Dir because images are on a different drive compared to all other files
#masterDir = '/media/fraserw/Thumber/SEP2017'
bf = '/media/fraserw/rocketdata/FEB2018/blinks/blink.57098'
if doFast:
    bf = '/media/fraserw/rocketdata/FEB2018/fast_blinks/blink_fast.57098'
elif doVeryFast:
    bf = '/media/fraserw/rocketdata/FEB2018/veryfast_blinks/blink_veryfast.57098'
if len(sys.argv)>1:
    bf = bf.replace('57098',sys.argv[1])



with open(bf,'rb') as han:
    mover_details = pickle.load(han)



candFile = candidatesPath + bf.split('/')[-1].replace('blink','/candidates')


cornersFile = masterDir+'/corners.txt'
with open(cornersFile) as han:
    data = han.readlines()

corners = {}
for i in range(len(data)):
    s = data[i].split()
    a = []
    for j in range(4):
        a.append([float(s[2*j+2]),float(s[2*j+3])])
    corners[s[0]] = [float(s[1]),np.array(a)]


#check for triplets that are fully contained in larger sets.
contained = []
for i in range(len(mover_details)):

    if len(mover_details[i][2:])>3:
        md = []
        for k in range(2,len(mover_details[i])):
            md.append(mover_details[i][k][1]+str(mover_details[i][k][2])+str(mover_details[i][k][3]))
        for j in range(len(mover_details)):
            if i != j:
                if len(mover_details[j])<len(mover_details[i]):
                    n = 0
                    for k in range(2,len(mover_details[j])):
                        mdk = mover_details[j][k][1]+str(mover_details[j][k][2])+str(mover_details[j][k][3])
                        if mdk in md:
                            n+=1
                    if n == len(mover_details[j][2:]):
                        contained.append(j)

print(len(mover_details))
print(len(contained))

candidates = []
for mi in range(  len(mover_details)):
    print(mi+1,len(mover_details))
    if mi in contained:
        continue



    xs = []
    ys = []
    ims = []
    snrs = []
    dist_low = mover_details[mi][0]
    dist_high = mover_details[mi][1]
    for i in range(2,len(mover_details[mi])):
        ims.append(mover_details[mi][i][1].replace('sex_save','fits'))
        xs.append(mover_details[mi][i][2])
        ys.append(mover_details[mi][i][3])
        snrs.append(mover_details[mi][i][4])



    fig = pyl.figure('Blinky',figsize=(26,13))

    possDrawing = False
    possCounter = 0
    possIms = []
    fieldCuts = []


    cutouts = []
    normers = []
    normed = []
    wcses = []
    X = []
    Y = []
    datas = []
    headers = []
    jds = []
    ras = []
    decs = []
    for i,im in enumerate(ims):
        #print(masterDir+'/*/'+im,xs[i],ys[i])
        fn = glob.glob(masterDir+'/*/HSC-R2/corr/'+im)[0]
        with fits.open(fn) as han:
            data = han[1].data
            header0 = han[0].header
            header = han[1].header
        ims[i] = fn
        datas.append(data)
        wcses.append(wcs.WCS(header))
        jds.append(header0['MJD'])
        print("{} {:>7.2f} {:>7.2f} {:>6.1f}".format(fn,xs[i],ys[i],snrs[i]))


    #require more than a 2 night span for the Feb 2018 data
    #but always look at the links that have 8 or more images regardless of jd range
    jds = np.array(jds)
    u_jd_int = np.unique(jds.astype('int'))
    print()
    if len(ims)<8:
        if (np.max(jds)-np.min(jds)<2 or len(u_jd_int)<3) and (doFast or doVeryFast):
            continue
    ########


    subplots = []
    if len(ims) == 3 or len(ims) == 4:
        nsp = 2
        subplots.append( pyl.subplot2grid((nsp, nsp+2), (0, 0), rowspan = 2, colspan = 2))
        subplots.append( pyl.subplot2grid((nsp, nsp+2), (0, 2), aspect='equal') )
        subplots.append( pyl.subplot2grid((nsp, nsp+2), (0, 3), aspect='equal') )
        subplots.append( pyl.subplot2grid((nsp, nsp+2), (1, 2), aspect='equal') )
        if len(ims) == 4:
            subplots.append( pyl.subplot2grid((nsp, nsp+2), (1, 3), aspect='equal') )
    elif len(ims) == 5 or len(ims) == 6:
        nsp = 3
        subplots.append( pyl.subplot2grid((nsp, nsp+2), (0, 0), rowspan = 3, colspan = 3))
        subplots.append( pyl.subplot2grid((nsp, nsp+2), (0, 3), aspect='equal') )
        subplots.append( pyl.subplot2grid((nsp, nsp+2), (0, 4), aspect='equal') )
        subplots.append( pyl.subplot2grid((nsp, nsp+2), (1, 3), aspect='equal') )
        subplots.append( pyl.subplot2grid((nsp, nsp+2), (1, 4), aspect='equal') )
        subplots.append( pyl.subplot2grid((nsp, nsp+2), (2, 3), aspect='equal') )
        if len(ims) == 6:
            subplots.append( pyl.subplot2grid((nsp, nsp+2), (2, 4), aspect='equal') )

    elif len(ims) == 7 or len(ims) == 8 or len(ims) == 9:
        nsp = 3
        subplots.append( pyl.subplot2grid((nsp, nsp+3), (0, 0), rowspan = 3, colspan = 3))
        subplots.append( pyl.subplot2grid((nsp, nsp+3), (0, 3), aspect='equal') )
        subplots.append( pyl.subplot2grid((nsp, nsp+3), (0, 4), aspect='equal') )
        subplots.append( pyl.subplot2grid((nsp, nsp+3), (0, 5), aspect='equal') )
        subplots.append( pyl.subplot2grid((nsp, nsp+3), (1, 3), aspect='equal') )
        subplots.append( pyl.subplot2grid((nsp, nsp+3), (1, 4), aspect='equal') )
        subplots.append( pyl.subplot2grid((nsp, nsp+3), (1, 5), aspect='equal') )
        subplots.append( pyl.subplot2grid((nsp, nsp+3), (2, 3), aspect='equal') )
        if len(ims) == 8 or len(ims) == 9:
            subplots.append( pyl.subplot2grid((nsp, nsp+3), (2, 4), aspect='equal') )
        if len(ims) == 9:
            subplots.append( pyl.subplot2grid((nsp, nsp+3), (2, 5), aspect='equal') )


    elif len(ims)>9:

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


    for i,im in enumerate(ims):
        X.append([])
        Y.append([])

        for j in range(len(wcses)):
            (ra,dec) = wcses[j].all_pix2world(xs[j],ys[j],0)
            if i==0:
                ras.append(ra)
                decs.append(dec)
            if i == j:
                X[-1].append(xs[i])
                Y[-1].append(ys[i])
            else:

                (xx,yy) = wcses[i].all_world2pix(ra,dec,0)
                X[-1].append(xx)
                Y[-1].append(yy)
    X = np.array(X)
    Y = np.array(Y)

    i = 0
    cut_width = widths[i]
    while (np.max(X[0,:])-np.min(X[0,:]))>cut_width or (np.max(Y[0,:])-np.min(Y[0,:]))>cut_width:
        cut_width = widths[i]
        i+=1

    for i,im in enumerate(ims):
        #print(np.median(datas[i][1850:2300,650:1390]))
        (A,B) = datas[i].shape
        bgf = bgFinder.bgFinder(datas[i][30:A-30:10,30:B-30:10])
        bg = bgf.fraserMode()

        big = np.zeros((A+cut_width*4+1,B+cut_width*4+1)).astype('float64')+bg
        big[cut_width*2:A+cut_width*2,cut_width*2:cut_width*2+B] = datas[i]
        cut = big[int(Y[i][0])+cut_width:int(Y[i][0])+3*cut_width,int(X[i][0])+cut_width:int(X[i][0])+3*cut_width]
        cutouts.append(np.copy(cut))


        if len(cut) == 0:
            normers.append([])
            normed.append(np.zeros((2*cut_width+1,2*cut_width+1)))
        else:
            (z1,z2) = numdisplay.zscale.zscale(cutouts[-1], contrast = contrast)
            normers.append( interval.ManualInterval(z1,z2))
            normed.append(normers[-1](cutouts[-1]))




    #findBGImage(np.array([ras[1],decs[1]]),jds[1],corners)
    #exit()


    #subplots[0].imshow(normed[0])
    #subplots[0].plot([cut_width-20,cut_width-5],[cut_width,cut_width],'r-')
    #subplots[0].plot([cut_width+5,cut_width+20],[cut_width,cut_width],'r-')
    #ubplots[0].plot([cut_width,cut_width],[cut_width-20,cut_width-5],'r-')
    #subplots[0].plot([cut_width,cut_width],[cut_width+5,cut_width+20],'r-')
    #subplots[0].set_ylim(subplots[0].get_ylim()[::-1])

    distantCandidate = False
    counter = 0
    draw(subplots,normed,counter,xs,ys,X,Y,cut_width)

    patches = []
    for i in range(len(cutouts)):
        subplots[i+1].imshow(normed[i])
        subplots[i+1].set_ylim(subplots[i+1].get_ylim()[::-1])
        #print(cut_width+(xs[i]-X[0][i]),cut_width+(ys[i]-Y[0][i]))
        patches.append(Circle((cut_width+(xs[i]-X[i][0]),cut_width+(ys[i]-Y[i][0])),radius=30, lw = 1.5,edgecolor='r',facecolor='none'))
        subplots[i+1].add_patch(patches[-1])
    #exit()
    patches[0].set_edgecolor('y')
    cid = fig.canvas.mpl_connect('key_press_event', blinky)

    pyl.show()

    if distantCandidate:
        candidates.append(mi)
        with open(candFile,'w+') as han:
            for i in candidates:
                han.write(str(i)+'\n')

with open(candFile,'w+') as han:
    for i in candidates:
        han.write(str(i)+'\n')

bn = candFile.split('.')[-1]
candCutsDir = candidatesPath+'/'+str(bn)
if not path.exists(candCutsDir):
    os.mkdir(candCutsDir)

for k,mi in enumerate(candidates):

    if not path.exists(candCutsDir+'/'+str(mi)):
        os.mkdir(candCutsDir+'/'+str(mi))
    outhan = open(candCutsDir+'/'+str(mi)+'/'+str(mi)+'.info','w+')

    xs = []
    ys = []
    ims = []
    snrs = []
    dist_low = mover_details[mi][0]
    dist_high = mover_details[mi][1]
    for i in range(2,len(mover_details[mi])):
        ims.append(mover_details[mi][i][1].replace('sex_save','fits'))
        xs.append(mover_details[mi][i][2])
        ys.append(mover_details[mi][i][3])
        snrs.append(mover_details[mi][i][4])

    print(ims)

    cutouts = []
    datas = []
    jds = []
    ras = []
    decs = []
    wcses = []
    for i,im in enumerate(ims):
        #print(masterDir+'/*/'+im,xs[i],ys[i])
        fn = glob.glob(masterDir+'/*/HSC-R2/corr/'+im)[0]
        with fits.open(fn) as han:
            data = han[1].data
            header0 = han[0].header
            header = han[1].header
        datas.append(data)
        ims[i] = fn
        wcses.append(wcs.WCS(header))
        jds.append(header0['MJD'])

    for i,im in enumerate(ims):
        (ra,dec) = wcses[i].all_pix2world(xs[i],ys[i],0)
        ras.append(ra)
        decs.append(dec)

        #print(np.median(datas[i][1850:2300,650:1390]))
        (A,B) = datas[i].shape
        bgf = bgFinder.bgFinder(datas[i][30:A-30:10,30:B-30:10])
        bg = bgf.fraserMode()

        big = np.zeros((A+cut_width*4+1,B+cut_width*4+1)).astype('float64')+bg
        big[cut_width*2:A+cut_width*2,cut_width*2:cut_width*2+B] = datas[i]
        cut = big[int(ys[i])+cut_width:int(ys[i])+3*cut_width,int(xs[i])+cut_width:int(xs[i])+3*cut_width]
        cutouts.append(np.copy(cut))


        entry = "{} {:>12.6f} {:>11.6f} {:>11.6f}".format(ims[i],jds[i],float(ras[i]),float(decs[i]))
        #entry = "{} {:>12.6f} {:>11.6f} {:>11.6f}".format(ims[i],jds[i],ras[i],decs[i])
        print(entry)
        outhan.write(entry+'\n')
    outhan.close()

    for i in range(len(ims)):
        fits.writeto(candCutsDir+'/'+str(mi)+'/'+str(i)+'.fits', cutouts[i], overwrite=True)
