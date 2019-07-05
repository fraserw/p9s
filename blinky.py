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

def draw(subplots,normed,counter,xs,ys,X,Y,cut_widthx,cut_widthy,fieldCut = None,fmt='r-'):
    subplots[0].cla()
    if fieldCut is None:
        subplots[0].imshow(normed[counter])
    else:
        subplots[0].imshow(fieldCut)
    (xa,xb) = subplots[0].get_xlim()
    (ya,yb) = subplots[0].get_ylim()
    if not doCTIO:
        subplots[0].set_xlim(xa,xb/2.0+100.0)
    else:
        subplots[0].set_xlim((xa+xb)/2-100.0,xb)
    (ya,yb) = subplots[0].get_ylim()
    diff_x = xb/2.0-xa+100.0
    mid_y = (yb+ya)/2
    #subplots[0].set_ylim(mid_y+diff_x/2.0,mid_y-diff_x/2.0)

    drawReticle(subplots[0],cut_widthx+xs[counter]-X[counter][0],cut_widthy+ys[counter]-Y[counter][0],fmt)
    subplots[0].set_ylim(subplots[0].get_ylim()[::-1])

    for i in range(len(X)):
        if i != counter:
            subplots[0].scatter(X[counter][i]-X[counter][0] + cut_widthx, Y[counter][i]-Y[counter][0] + cut_widthy, marker = 'o', edgecolor = 'r', facecolor = 'none', s =600)
            #subplots[0].scatter((X[0][i]-xs[0]) + cut_width, (Y[0][i]-ys[0]) + cut_width, marker = 'o', edgecolor = 'r', facecolor = 'none',s=400)


def blinky(event):
    global subplots,normed, patches, counter,nsp,ims,xs,ys,ras,decs,jds,X,Y,dist_low,dist_high,distantCandidate,corners, cut_widthx, cut_widthy, possCounter, possIms, possDrawing, fieldCuts, contrast

    if event.key == 'q':
        exit()

    if event.key in ['a','tab','d']:
        possIms = []
        possDrawing = False
        possCounter = 0
        for ii in range(len(patches)):
            if len(subplots)>1:
                subplots[ii+1].patches[-1].set_edgecolor('r')

    if event.key in ['a','tab']:
        counter += 1
        counter = counter%len(normed)
        if len(subplots)>1:
            subplots[counter+1].patches[-1].set_edgecolor('y')
    elif event.key in ['d']:
        counter -= 1
        counter = counter%len(normed)
        if len(subplots)>1:
            subplots[counter+1].patches[-1].set_edgecolor('y')

    if event.key in['v']:
        comm = 'ds9 '
        for ii in range(len(ims)):
            comm+='{} -pan to {} {} -regions command "circle({},{},10)" '.format(ims[ii],xs[ii],ys[ii],xs[ii],ys[ii])
        print(comm)

    if event.key in ['a','tab','d']:
        draw(subplots,normed,counter,xs,ys,X,Y,cut_widthx,cut_widthy)
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
                big = np.zeros((A+cut_widthy*4+1,B+cut_widthx*4+1)).astype('float64')+bg
                big[cut_widthy*2:A+cut_widthy*2,cut_widthx*2:cut_widthx*2+B] = fieldData
                cut = big[int(yy)+cut_widthy:int(yy)+3*cut_widthy,int(xx)+cut_widthx:int(xx)+3*cut_widthx]


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
            draw(subplots,normed,counter,xs,ys,X,Y,cut_widthx,cut_widthy,fieldCut = fieldCut,fmt='w-')

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

widths = [200,300,500,700,900,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000]
contrast = 0.4

doFast = False
doVeryFast = False
doCTIO = False
if '--fast' in sys.argv:
    doFast = True
    print('Using fast_bricks')
elif '--veryfast' in sys.argv:
    doVeryFast = True
    print('Using veryfast_bricks')
elif '--ctio' in sys.argv:
    doCTIO = True

if doCTIO:
    masterDir = '/media/fraserw/rocketdata/CTIO_DEC_2018'

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
bf = masterDir+'/blinks/blink.57098'
if doFast:
    bf = masterDir+'/fast_blinks/blink_fast.57098'
elif doVeryFast:
    bf = masterDir+'/veryfast_blinks/blink_veryfast.57098'
if len(sys.argv)>1:
    bf = bf.replace('57098',sys.argv[1])



with open(bf,'rb') as han:
    mover_details = pickle.load(han)

if '--skipFastCands' in sys.argv:
    with open(bf.replace('very',''),'rb') as han:
        movers_to_skip = pickle.load(han)

    for i in range(len(mover_details)):
        for j in range(len(movers_to_skip)):
            if len(mover_details[i]) == len(movers_to_skip[j]):
                haves = 0
                for k in range(2,len(mover_details[i])):
                    have = False
                    for l in range(2,len(movers_to_skip[j])):
                        if movers_to_skip[j][l][1] == mover_details[i][k][1] and movers_to_skip[j][l][2] == mover_details[i][k][2] and  movers_to_skip[j][l][3] == mover_details[i][k][3] and movers_to_skip[j][l][4] == mover_details[i][k][4]:
                            have = True
                    #print(i,j,have)
                    if have:
                        haves+=1
                if len(mover_details[i])-2 == haves:
                    print('Skipping',j)
                    exit()


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
    if len(mover_details[i][2:])>3 and i not in contained:
        md = []
        for k in range(2,len(mover_details[i])):
            md.append(mover_details[i][k][1]+str(mover_details[i][k][2])+str(mover_details[i][k][3]))
        for j in range(len(mover_details)):
            if i != j:
                if len(mover_details[j])<=len(mover_details[i]):
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
    title = str(mi+1)+'/'+str(len(mover_details))
    if mi in contained:
        print(mi+1,len(mover_details),'Contained')
        continue
    else:
        print(mi+1,len(mover_details))



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



    fig = pyl.figure('Blinky',figsize=(28,13))
    fig.patch.set_facecolor('0.6')

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
        #fn = glob.glob(masterDir+'/*/HSC-R2/corr/'+im)[0]
        if not doCTIO:
            fn = (glob.glob('/media/fraserw/Hammer/DEC2018/*/HSC-R2/corr/'+im)+glob.glob('/media/fraserw/Thumber/DEC2018_also/*/HSC-R2/corr/'+im))[0]
        else:
            fn = glob.glob('/media/fraserw/Hammer/CTIO_DEC_2018/NH_?/'+im)[0]
        with fits.open(fn) as han:
            if not doCTIO:
                data = han[1].data
                header0 = han[0].header
                header = han[1].header
            else:
                data = han[0].data
                header0 = han[0].header
                header = han[0].header
                header0['MJD'] = header['MJD-OBS']

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
        print(np.max(jds)-np.min(jds), len(u_jd_int))
        if (np.max(jds)-np.min(jds)<0.6 or len(u_jd_int)<2) and (doFast or doVeryFast):
            print(np.max(jds)-np.min(jds), len(u_jd_int))
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

    if '--veryfast' in sys.argv:# or '--fast' in sys.argv:
        print('Not drawing individual frames.')
        subplots = []
        subplots.append( pyl.subplot2grid((1,1), (0, 0), rowspan = 1, colspan = 1))

    subplots[0].set_xlabel(title)

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
    cut_widthx = widths[i]
    cut_widthy = widths[i]
    while (np.max(X[0,:])-np.min(X[0,:]))>cut_widthx:
        cut_widthx = widths[i]
        i+=1
    i = 0
    while (np.max(Y[0,:])-np.min(Y[0,:]))>cut_widthy:
        cut_widthy = widths[i]
        i+=1
    cut_widthx+=50
    cut_widthy+=50

    for i,im in enumerate(ims):
        #print(np.median(datas[i][1850:2300,650:1390]))
        (A,B) = datas[i].shape
        bgf = bgFinder.bgFinder(datas[i][30:A-30:10,30:B-30:10])
        bg = bgf.fraserMode()

        big = np.zeros((A+cut_widthy*4+1,B+cut_widthx*4+1)).astype('float64')+bg
        big[cut_widthy*2:A+cut_widthy*2,cut_widthx*2:cut_widthx*2+B] = datas[i]
        cut = big[int(Y[i][0])+cut_widthy:int(Y[i][0])+3*cut_widthy,int(X[i][0])+cut_widthx:int(X[i][0])+3*cut_widthx]
        #cut = big[int(meanPos[1])+cut_width:int(meanPos[1])+3*cut_width,int(meanPos[0])+cut_width:int(meanPos[0])+3*cut_width]
        cutouts.append(np.copy(cut))


        if len(cut) == 0:
            normers.append([])
            normed.append(np.zeros((2*cut_widthy+1,2*cut_widthx+1)))
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
    draw(subplots,normed,counter,xs,ys,X,Y,cut_widthx,cut_widthy)

    patches = []
    for i in range(len(cutouts)):
        if len(subplots)>1:
            subplots[i+1].imshow(normed[i])
            subplots[i+1].set_ylim(subplots[i+1].get_ylim()[::-1])
            #print(cut_width+(xs[i]-X[0][i]),cut_width+(ys[i]-Y[0][i]))
        patches.append(Circle((cut_widthx+(xs[i]-X[i][0]),cut_widthy+(ys[i]-Y[i][0])),radius=30, lw = 1.5,edgecolor='r',facecolor='none'))
        if len(subplots)>1:
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

print(bf)
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
        #fn = glob.glob(masterDir+'/*/HSC-R2/corr/'+im)[0]
        fn = (glob.glob('/media/fraserw/Hammer/DEC2018/*/HSC-R2/corr/'+im)+glob.glob('/media/fraserw/Thumber/DEC2018_also/*/HSC-R2/corr/'+im))[0]
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

        big = np.zeros((A+cut_widthy*4+1,B+cut_widthx*4+1)).astype('float64')+bg
        big[cut_widthy*2:A+cut_widthy*2,cut_widthx*2:cut_widthx*2+B] = datas[i]
        cut = big[int(ys[i])+cut_widthy:int(ys[i])+3*cut_widthy,int(xs[i])+cut_widthx:int(xs[i])+3*cut_widthx]
        cutouts.append(np.copy(cut))


        entry = "{} {:>12.6f} {:>11.6f} {:>11.6f}".format(ims[i],jds[i],float(ras[i]),float(decs[i]))
        #entry = "{} {:>12.6f} {:>11.6f} {:>11.6f}".format(ims[i],jds[i],ras[i],decs[i])
        print(entry)
        outhan.write(entry+'\n')
    outhan.close()

    for i in range(len(ims)):
        fits.writeto(candCutsDir+'/'+str(mi)+'/'+str(i)+'.fits', cutouts[i], overwrite=True)
print()
