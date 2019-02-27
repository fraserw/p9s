
import numpy as np,pylab as pyl
import pickle

from astropy.coordinates import SkyCoord
from astropy import units as u

from catObj import *
from paths import *
from p9makebricks import getBrickNum
from p9gettransients import gc_dist
import earth


def idl_hashtag_incorrect_interpretation(big_arr,xhat,yhat,zhat,i):
    xe = np.copy(big_arr)
    xe[0] *= xhat[0]
    xe[1] *= xhat[1]
    xe[2] *= xhat[2]
    ye = np.copy(big_arr)
    ye[0] *= yhat[0]
    ye[1] *= yhat[1]
    ye[2] *= yhat[2]
    ze = np.copy(big_arr)
    ze[0] *= zhat[0]
    ze[1] *= zhat[1]
    ze[2] *= zhat[2]

    xe[0]-=xe[i][0]
    xe[1]-=xe[i][1]
    xe[2]-=xe[i][2]
    ye[0]-=ye[i][0]
    ye[1]-=ye[i][1]
    ye[2]-=ye[i][2]
    ze[0]-=ze[i][0]
    ze[1]-=ze[i][1]
    ze[2]-=ze[i][2]

    return (xe,ye,ze)
def idl_hashtag(big_arr,xhat,yhat,zhat,i):
    xe = big_arr[:,0]*xhat[0] + big_arr[:,1]*xhat[1] + big_arr[:,2]*xhat[2]
    ye = big_arr[:,0]*yhat[0] + big_arr[:,1]*yhat[1] + big_arr[:,2]*yhat[2]
    ze = big_arr[:,0]*zhat[0] + big_arr[:,1]*zhat[1] + big_arr[:,2]*zhat[2]

    xe -= xe[i]
    ye -= ye[i]
    ze -= ze[i]
    return (xe,ye,ze)



if __name__ == "__main__":
    import glob,os,sys,pickle
    from os import path
    import time as Time

    if '--veryfast' in sys.argv:
        maxdays = 9.0 #max separation between centre and final/initial day
        slow_time = 0.5#12 #time in hours to look for motion
        fastestp9 = 6.0 #arcsec/hr of fastest p9

        min_dist = 20.0 #Minimum distance to probe
        max_dist = 70.0
        dist_step = 0.1

        n_min_det = 4

        showDS9string = True

        pathString = 'veryfast_blinks'

        print('Searching for very fast movers!')


    elif '--fast' in sys.argv:
        maxdays = 9.0 #max separation between centre and final/initial day
        slow_time = 0.5#12 #time in hours to look for motion
        fastestp9 = 3.0 #arcsec/hr of fastest p9

        min_dist = 60.0 #Minimum distance to probe
        max_dist = 250.0
        dist_step = 0.1

        n_min_det = 4

        showDS9string = True

        pathString = 'fast_blinks'

        print('Searching for fast movers!')

    else:
        maxdays = 9.0 #max separation between centre and final/initial day
        slow_time = 1.0#12 #time in hours to look for motion
        fastestp9 = 0.5 #arcsec/hr of fastest p9
        #fastestobj = 9.0 #fastest object in arcsec/hr

        min_dist = 200.0 #Minimum distance to probe
        max_dist = 1500.0
        dist_step = 0.5

        n_min_det = 3

        pathString = 'blinks'

        showDS9string = False



    #distance range to test
    dist_range = np.arange(min_dist,max_dist,dist_step)  #(200,1500,1)
    gamtest = 1./dist_range

    #min number of detections to call something a mover


    Earth = earth.Earth()

    #masterDir = '/media/fraserw/rocketdata/SEP2017'

    if len(sys.argv)>2:
        if '--diff' in sys.argv:
            masterDir += '/DiffCatalog'
        else:
            print('Were you trying to use the diff catalog? If so, pass --diff.\n')
            #exit()

    blinksPath = masterDir+'/'+pathString

    if not path.exists(blinksPath):
        os.mkdir(blinksPath)
    #masterDir = '/media/fraserw/rocketdata/scratch'

    brick_files = glob.glob(masterDir+'/bricks/*brick')
    brick_files.sort()




    bn_i = 0
    if len(sys.argv)>1:
        bn_i = int(float(sys.argv[1]))

    #for i in range(len(brick_files)):
    #    if '92116' in brick_files[i]:
    #        print(i,brick_files[i])
    #        bn_i = i
    #exit()

    print(brick_files[bn_i])
    with open(brick_files[bn_i],'rb') as han:
        brick = pickle.load(han)

    bn = int(float(brick_files[bn_i].split('/')[-1].split('.')[0]))
    blinkfn = blinksPath+'/blink.'+str(bn)
    if '--fast' in sys.argv:
        blinkfn = blinksPath+'/blink_fast.'+str(bn)
    elif '--veryfast' in sys.argv:
        blinkfn = blinksPath+'/blink_veryfast.'+str(bn)



    #print(brick.ra)
    #exit()
    trans = brickObj()
    w = np.where(brick.wf)
    trans.x = brick.x[w]
    trans.y = brick.y[w]
    trans.ra = brick.ra[w]
    trans.dec = brick.dec[w]
    trans.flux = brick.flux[w]
    trans.snr = brick.snr[w]
    trans.mag = brick.mag[w]
    trans.fwhm_image = brick.fwhm_image[w]
    trans.jd = brick.jd[w]
    trans.seeing = brick.seeing[w]
    trans.ellip = brick.ellip[w]
    trans.astrms = brick.astrms[w]
    trans.images = brick.images[w]
    trans.wf = np.array(brick.wf)[w]
    trans.ws = np.array(brick.ws)[w]
    #trans.wstat = np.array(brick.wstat)[w]
    for j in range(len(trans.x)):
        trans.brick.append(bn)


    min_ra = np.min(trans.ra)
    min_dec = np.min(trans.dec)
    max_ra = np.max(trans.ra)
    max_dec = np.max(trans.dec)


    nts = len(trans.x)
    br  = np.zeros(nts)+bn
    mxd = fastestp9*maxdays*24.0/3600.0
    mxr = mxd/np.cos(np.median(trans.dec)*d2r)


    bb = np.concatenate([getBrickNum(trans.ra+mxr,trans.dec)[0],
                        getBrickNum(trans.ra-mxr,trans.dec)[0],
                        getBrickNum(trans.ra+mxr,trans.dec+mxd)[0],
                        getBrickNum(trans.ra+mxr,trans.dec-mxd)[0],
                        getBrickNum(trans.ra-mxr,trans.dec-mxd)[0],
                        getBrickNum(trans.ra-mxr,trans.dec+mxd)[0],
                        getBrickNum(trans.ra,trans.dec-mxd)[0],
                        getBrickNum(trans.ra,trans.dec+mxd)[0]])
    bb_unique = np.sort(np.unique(bb))

    ##########
    #print("!!!!!!!!!remove this line in the code below before production runs!!!!!!!!!!!!")
    #bb_unique = []
    ###########

    for bb in bb_unique:

        if bb == bn: continue

        extra_brick_file = brick_files[bn_i].replace(str(bn),str(bb))
        if extra_brick_file in brick_files:
            with open(extra_brick_file,'rb') as han:
                extra_brick = pickle.load(han)

            w = np.where( (extra_brick.wf) & (extra_brick.ra>min_ra-mxr) & (extra_brick.ra<max_ra+mxr) & (extra_brick.dec>min_dec-mxd) & (extra_brick.dec<max_dec+mxd))

            trans.x = np.concatenate([trans.x,extra_brick.x[w]])

            trans.y = np.concatenate([trans.y,extra_brick.y[w]])
            trans.ra = np.concatenate([trans.ra,extra_brick.ra[w]])
            trans.dec = np.concatenate([trans.dec,extra_brick.dec[w]])
            trans.flux = np.concatenate([trans.flux,extra_brick.flux[w]])
            trans.snr = np.concatenate([trans.snr,extra_brick.snr[w]])
            trans.mag = np.concatenate([trans.mag,extra_brick.mag[w]])
            trans.fwhm_image = np.concatenate([trans.fwhm_image,extra_brick.fwhm_image[w]])
            trans.jd = np.concatenate([trans.jd,extra_brick.jd[w]])
            trans.seeing = np.concatenate([trans.seeing,extra_brick.seeing[w]])
            trans.ellip = np.concatenate([trans.ellip,extra_brick.ellip[w]])
            trans.astrms = np.concatenate([trans.astrms,extra_brick.astrms[w]])
            trans.images = np.concatenate([trans.images,extra_brick.images[w]])
            trans.wf = np.concatenate([trans.wf,extra_brick.wf[w]])
            trans.ws = np.concatenate([trans.ws,extra_brick.ws[w]])
            #trans.wstat = np.concatenate([trans.wstat,extra_brick.wstat[w]])
            for j in range(len(w[0])):
                trans.brick.append(bb)
    trans.brick = np.array(trans.brick)


    add_test_mover = False
    if add_test_mover:
        import genFakeObject

        fake_rate_ra = -0.37 #"/hr << 0.28 is roughly 500 AU opposition
        fake_rate_dec = -0.22
        jd_unique = np.sort(np.unique(trans.jd))

        fake_t = [jd_unique[0]]
        fake_ra = [trans.ra[0]+0.001]
        fake_dec = [trans.dec[0]]
        for j in range(len(jd_unique)):
            if (jd_unique[j]-fake_t[-1])*24>slow_time:
                fake_t.append(jd_unique[j])
        for j in range(1,len(fake_t)):
            fake_ra.append(fake_ra[0]+(jd_unique[j]-fake_t[0])*24*fake_rate_ra/3600.0)
            fake_dec.append(fake_dec[0]+(jd_unique[j]-fake_t[0])*24*fake_rate_dec/3600.0)

        useEphem = True
        if useEphem:
            (fake_ra,fake_dec) = genFakeObject.getradec(fake_t)
        print('Adding fake object')
        print('   ',fake_t)
        print('   ',fake_ra)
        print('   ',fake_dec)
        print
        #
        #exit()

        trans.ra = np.concatenate([trans.ra,np.array(fake_ra)])
        trans.dec = np.concatenate([trans.dec,np.array(fake_dec)])
        trans.jd = np.concatenate([trans.jd,np.array(fake_t)])
        trans.x = np.concatenate([trans.x,np.zeros(len(fake_t))])
        trans.y = np.concatenate([trans.y,np.zeros(len(fake_t))])
        trans.flux = np.concatenate([trans.flux,np.zeros(len(fake_t))+1000.0])
        trans.snr = np.concatenate([trans.snr,np.zeros(len(fake_t))+100.0])
        trans.mag = np.concatenate([trans.mag,np.zeros(len(fake_t))])
        trans.fwhm_image = np.concatenate([trans.fwhm_image,np.zeros(len(fake_t))])
        trans.seeing = np.concatenate([trans.seeing,np.zeros(len(fake_t))])
        trans.ellip = np.concatenate([trans.ellip,np.zeros(len(fake_t))])
        trans.astrms = np.concatenate([trans.astrms,np.zeros(len(fake_t))])
        trans.images = np.concatenate([trans.images,np.zeros(len(fake_t))])
        trans.wf = np.concatenate([trans.wf,np.zeros(len(fake_t))])
        trans.ws = np.concatenate([trans.ws,np.zeros(len(fake_t))])
        #trans.wstat = np.concatenate([trans.wstat,np.zeros(len(fake_t))])
        trans.brick = np.concatenate([trans.brick,np.zeros(len(fake_t))+bn])



    args = np.argsort(trans.jd)
    trans.x = trans.x[args]
    trans.y = trans.y[args]
    trans.ra = trans.ra[args]
    trans.dec = trans.dec[args]
    trans.flux = trans.flux[args]
    trans.snr = trans.snr[args]
    trans.mag = trans.mag[args]
    trans.fwhm_image = trans.fwhm_image[args]
    trans.jd = trans.jd[args]
    trans.seeing = trans.seeing[args]
    trans.ellip = trans.ellip[args]
    trans.astrms = trans.astrms[args]
    trans.images = trans.images[args]
    trans.wf = trans.wf[args]
    trans.ws = trans.ws[args]
    #trans.wstat = trans.wstat[args]
    trans.brick = trans.brick[args]

    dmag = 1.09/trans.snr




    tl, tb = [], []
    coords = SkyCoord(trans.ra*u.deg,trans.dec*u.deg,frame = 'icrs')
    ecl_coords = np.array(coords.transform_to('barycentrictrueecliptic'))
    for j in range(len(ecl_coords)):
        tl.append(ecl_coords[j].lon.degree)
        tb.append(ecl_coords[j].lat.degree)
    #step = 1000
    #i = 0
    #while i<len(trans.ra):
    #
    #    t1=Time.time()
    #    coords = SkyCoord(trans.ra[i:i+step]*u.deg,trans.dec[i:i+step]*u.deg,frame = 'icrs')
    #    ecl_coords = np.array(coords.transform_to('barycentrictrueecliptic'))
    #    for j in range(len(ecl_coords)):
    #        tl.append(ecl_coords[j].lon.degree)
    #        tb.append(ecl_coords[j].lat.degree)
    #    print(i,len(trans.ra),Time.time()-t1)
    #    i+=step
    tl = np.array(tl)*d2r
    tb = np.array(tb)*d2r


    used = np.zeros(len(tl)) #10 when used, otherwise some other value

    #and because mike inexplicably changes variable names...
    #obj = trans


    earth_coords = Earth(trans.jd)
    big_arr = np.zeros((len(trans.jd),3)).astype('float64')
    big_arr[:,0] = earth_coords[:,0]
    big_arr[:,1] = earth_coords[:,1]



    time = (trans.jd - np.min(trans.jd))*24.0 #time in hours

    ####
    #print(np.unique(time))
    #w = np.where((time>slow_time) & (trans.brick == bn))
    #print(w[0][0],w[0][-1])
    #w = np.where((time<time[-1]-slow_time) & (trans.brick == bn))
    #print(w[0][0],w[0][-1])
    #exit()
    ####

    try:
        strt = np.where((time>slow_time) & (trans.brick == bn))[0][0]
        stp = np.where((time<time[-1]-slow_time) & (trans.brick == bn))[0][-1]
    except:
        strt = -1
        stp = -1




    print(strt,stp)

    if add_test_mover:
        ################################3
        print('HACK')
        ind = 1
        w = np.where((trans.ra==fake_ra[ind])&(trans.dec==fake_dec[ind])&(trans.jd==fake_t[ind]))
        strt = w[0][0]
        stp = w[0][0]+1
        print(strt,stp)
        ######################################


    movers = []
    mover_details = []
    ndet = 0
    dis = 0.0
    move_idx = 0
    for i in range(strt,stp):
        if i in used: continue

        if i%100 == 0:
            print(i,stp)

        dback = np.max(time[i]-time)*fastestp9/3600.0
        dforward = np.max(time-time[i])*fastestp9/3600.0
        dmax = max(dforward,dback)

        mag_diff_lim = 1.3+dmag[i]

        #mike comment
        # I am only looking at this with HIGHER ra in the past.    xe = np.sum(big_arr[:,0]*xhat[:,0] + big_arr[:,1]*xhat[:,1] + big_arr[:,2]*xhat[:,2])

	    # implicitly assuming retrograde motion here!
        dra = trans.ra[i]-trans.ra
        ddec = trans.dec[i]-trans.dec
        w1 = np.where( (time[i] - time > slow_time) & (np.abs(ddec)<dback) & (np.abs(dra)*np.cos(trans.dec[i]*d2r) < dback) & (np.abs(trans.mag[i]-trans.mag)<mag_diff_lim) &(used<10))
        w2 = np.where( (time - time[i] > slow_time) & (np.abs(ddec)<dforward) & (np.abs(dra)*np.cos(trans.dec[i]*d2r) < dforward) & (np.abs(trans.mag[i]-trans.mag)<mag_diff_lim) & (used<10))



        if len(w1[0])>0 and len(w2[0])>0:
            d1 = gc_dist(trans.ra[i],trans.dec[i],trans.ra[w1],trans.dec[w1])*3600.0
            d2 = gc_dist(trans.ra[i],trans.dec[i],trans.ra[w2],trans.dec[w2])*3600.0

            w11 = np.where(d1/(time[i]-time[w1])<fastestp9)
            w22 = np.where(d2/(time[w2]-time[i])<fastestp9)

            if len(w11[0])>0 and len(w22[0])>0:

                l0 = tl[i]
                b0 = tb[i]

                zhat = np.array([ np.cos(l0)*np.cos(b0),  np.sin(l0)*np.cos(b0),np.sin(b0)]) #unit vectors in direction of l0,b0
                xhat = np.array([ -zhat[1],zhat[0],0.0])
                xhat /= np.sum(xhat*xhat)**0.5
                yhat = np.cross(zhat,xhat)

                #tangent plane coordinates for all observations
                l = tl[w1[0][w11]]
                b = tb[w1[0][w11]]
                x1 = np.cos(b)*np.sin(l-l0)/(np.sin(b0) * np.sin(b) + np.cos(b0)*np.cos(b)*np.cos(l-l0))
                y1 = (-np.sin(b0)*np.cos(b)*np.cos(l-l0) + np.sin(b)*np.cos(b0)) / (np.sin(b0)*np.sin(b) + np.cos(b0)*np.cos(b)*np.cos(l-l0))

                l = tl[w2[0][w22]]
                b = tb[w2[0][w22]]
                x3 = np.cos(b)*np.sin(l-l0)/(np.sin(b0) * np.sin(b) + np.cos(b0)*np.cos(b)*np.cos(l-l0))
                y3 = (-np.sin(b0)*np.cos(b)*np.cos(l-l0) + np.sin(b)*np.cos(b0)) / (np.sin(b0)*np.sin(b) + np.cos(b0)*np.cos(b)*np.cos(l-l0))


                bt1 = trans.jd[w1[0][w11]] - trans.jd[i]
                bt3 = trans.jd[w2[0][w22]] - trans.jd[i]

                (xe,ye,ze) = idl_hashtag(big_arr,xhat,yhat,zhat,i)


                xe1 = xe[w1[0][w11]]
                xe3 = xe[w2[0][w22]]
                ye1 = ye[w1[0][w11]]
                ye3 = ye[w2[0][w22]]

                n1 = len(w11[0])*len(w22[0])

                res = np.zeros((len(bt1),len(bt3),len(gamtest))).astype('float64')
                fes = np.zeros((len(bt1),len(bt3),len(gamtest))).astype('float64')
                for j,bt1val in enumerate(bt1):
                    for k,bt3val in enumerate(bt3):
                        """
                        #the code below is functional, maybe slow
                        resg = np.zeros(len(gamtest)).astype('float64')
                        fesg = np.zeros(len(gamtest)).astype('float64')
                        for g,gam in enumerate(gamtest):

                            adot = (np.sign(bt1val)*(x1[j]+xe1[j]*gam)+np.sign(bt3val)*(x3[k]+xe3[k]*gam))/(np.abs(bt1val) + np.abs(bt3val))
                            bdot = (np.sign(bt1val)*(y1[j]+ye1[j]*gam)+np.sign(bt3val)*(y3[k]+ye3[k]*gam))/(np.abs(bt1val) + np.abs(bt3val))

                            xp1 = bt1val*adot - gam*xe1[j]
                            yp1 = bt1val*bdot - gam*ye1[j]
                            xp3 = bt3val*adot - gam*xe3[k]
                            yp3 = bt3val*bdot - gam*ye3[k]

                            #calculate the residuals
                            rr = ( ((xp1-x1[j])**2 + (xp3-x3[k])**2 + (yp1-y1[j])**2 + (yp3-y3[k])**2)**0.5)/d2r*3600.0
                            f = (adot**2 + bdot**2)*(gam**(-3))*3389.0

                            #print(1./gam,rr,f,w1[0][w11][j],w2[0][w22][k])

                            resg[g] = rr
                            fesg[g] = f

                        res[j,k,:] = resg
                        fes[j,k,:] = fesg
                        #end slow section
                        """
                        #this code is an attempt to speed up the above

                        xe1mgam = xe1[j]*gamtest
                        ye1mgam = ye1[j]*gamtest
                        xe3mgam = xe3[k]*gamtest
                        ye3mgam = ye3[k]*gamtest
                        adot = (np.sign(bt1val)*(x1[j]+xe1mgam)+np.sign(bt3val)*(x3[k]+xe3mgam))/(np.abs(bt1val) + np.abs(bt3val))
                        bdot = (np.sign(bt1val)*(y1[j]+ye1mgam)+np.sign(bt3val)*(y3[k]+ye3mgam))/(np.abs(bt1val) + np.abs(bt3val))

                        xp1 = bt1val*adot - xe1mgam
                        yp1 = bt1val*bdot - ye1mgam
                        xp3 = bt3val*adot - xe3mgam
                        yp3 = bt3val*bdot - ye3mgam

                        #calculate the residuals
                        Resg = ( ((xp1-x1[j])**2 + (xp3-x3[k])**2 + (yp1-y1[j])**2 + (yp3-y3[k])**2)**0.5)/d2r*3600.0
                        Fesg = (adot**2 + bdot**2)*(gamtest**(-3))*3389.0

                        res[j,k,:] = Resg
                        fes[j,k,:] = Fesg
                        #end fast section


                w = np.where((res<1.0) & (fes<5.0))
                taken = []
                for j in range(len(w[0])):
                    ind_1 = w1[0][w11[0]][w[0][j]]
                    ind_3 = w2[0][w22[0]][w[1][j]]

                    if [ind_1,i,ind_3] not in taken:
                        taken.append([ind_1,i,ind_3])

                        test_mover_ind = np.array(taken[-1])
                        test_mover_dists = [dist_range[w[0][j]]]
                        for l in range(len(w[0])):
                            if ind_1 == w1[0][w11[0]][w[0][l]] and ind_3 == w2[0][w22[0]][w[1][l]] and l!=j:
                                test_mover_dists.append(dist_range[w[0][l]])

                        #print(test_mover_dists,'triplet')


                        used[test_mover_ind] = 5

                        w_add = np.where( (np.abs(time[i]-time)>0.008) & (time!=time[test_mover_ind[0]]) & (time!=time[test_mover_ind[2]]) & (np.abs(trans.dec[i]-trans.dec)<dmax) & (np.abs(trans.ra[i]-trans.ra)*np.cos(trans.dec[i]*d2r)<dmax) & (np.abs(trans.mag[i]-trans.mag)<mag_diff_lim) &(used<5) )
                        #print(i,w_add)
                        if len(w_add[0])>0: #we have additional candidates at different times to add to the test mover object
                            d = gc_dist(trans.ra[i],trans.dec[i],trans.ra[w_add],trans.dec[w_add])*3600.0
                            ww = np.where(d/np.abs(time[i]-time[w_add])<fastestp9)

                            w_add = w_add[0][ww] #objects potentially worth adding to the triplet

                            ###for now just loop through all the additional sources that we might be able to add
                            ###this loop is temporal. It may be better to loop over minimum residual, or closest distance
                            ###or something like that. But for now, just loop over time.

                            ###also for now just adopt the same index as the "middle point", rather than reset to the actual middle point.
                            ###this saves the effort of having to compute xhat and so on.
                            for ind in w_add:
                                test_mover_add_ind = np.concatenate([test_mover_ind,np.array([ind])])

                                l = tl[test_mover_add_ind]
                                b = tb[test_mover_add_ind]

                                x_add = np.cos(b)*np.sin(l-l0)/(np.sin(b0) * np.sin(b) + np.cos(b0)*np.cos(b)*np.cos(l-l0))
                                y_add = (-np.sin(b0)*np.cos(b)*np.cos(l-l0) + np.sin(b)*np.cos(b0)) / (np.sin(b0)*np.sin(b) + np.cos(b0)*np.cos(b)*np.cos(l-l0))

                                bt_add = trans.jd[test_mover_add_ind] - trans.jd[i]

                                #don't need to do this again because we are still using the same point for the reference of the tangent plane
                                #(xe,ye,ze) = idl_hashtag(big_arr,xhat,yhat,zhat,i)
                                xe_add = xe[test_mover_add_ind]
                                ye_add = ye[test_mover_add_ind]
                                ze_add = ze[test_mover_add_ind]


                                alreadyAdded = False
                                """
                                #slow but functional
                                for g,gam in enumerate(gamtest):
                                    adot_add = np.sum(np.sign(bt_add)*(x_add+xe_add*gam))/np.sum(np.abs(bt_add))
                                    bdot_add = np.sum(np.sign(bt_add)*(y_add+ye_add*gam))/np.sum(np.abs(bt_add))

                                    xp_add = bt_add*adot_add - gam*xe_add
                                    yp_add = bt_add*bdot_add - gam*ye_add

                                    #calculate the residuals
                                    rr = (np.sum((xp_add - x_add)**2 + (yp_add - y_add)**2)**0.5)*3600.0/d2r
                                    ff = (adot_add**2 + bdot_add**2)*(gam**(-3))*3389.0

                                    #print(1./gam,rr,f,test_mover_add_ind)
                                    #resg[g] = rr
                                    #fesg[g] = f
                                    if (rr<1 and ff<5):
                                        #print(1./gam,rr,ff,test_mover_add_ind[-1])
                                        if not alreadyAdded:
                                            test_mover_ind = np.copy(test_mover_add_ind)
                                            alreadyAdded = True
                                            test_mover_dists = []
                                        test_mover_dists.append( 1./gam)
                                #end slow section
                                """
                                #optimization of the above slow bit
                                rep_gamtest = np.repeat(np.array([gamtest]),len(xe_add)).reshape(len(gamtest),len(xe_add))
                                xemgam = xe_add*rep_gamtest
                                yemgam = ye_add*rep_gamtest


                                bt_add_sum = np.sum(np.abs(bt_add))
                                adot_add = np.sum(np.sign(bt_add)*(x_add+xemgam),axis=1)/bt_add_sum
                                bdot_add = np.sum(np.sign(bt_add)*(y_add+yemgam),axis=1)/bt_add_sum

                                rep_adot_add = np.repeat(np.array([adot_add]),len(bt_add)).reshape(len(gamtest),len(bt_add))
                                rep_bdot_add = np.repeat(np.array([bdot_add]),len(bt_add)).reshape(len(gamtest),len(bt_add))
                                xp_add = rep_adot_add*bt_add - xemgam
                                yp_add = rep_bdot_add*bt_add - yemgam


                                rr = (np.sum((xp_add - x_add)**2 + (yp_add - y_add)**2,axis=1)**0.5)*3600.0/d2r
                                ff = (adot_add**2 + bdot_add**2)*(gamtest**(-3))*3389.0


                                W_add = np.where((rr<1) & (ff<5))
                                if len(W_add[0])>0:
                                    if not alreadyAdded:
                                        alreadyAdded = True
                                        test_mover_ind = np.copy(test_mover_add_ind)
                                        test_mover_dists = 1./gamtest[W_add]
                                #end fast section


                        test_mover_dists = np.array(test_mover_dists)
                        if len(test_mover_ind)>=n_min_det:
                            movers.append(np.copy(np.sort(test_mover_ind)))

                            mover_details.append([np.min(test_mover_dists),np.max(test_mover_dists)])
                            used[test_mover_ind] = 10

                            print('\nMover!',i,test_mover_ind,np.min(test_mover_dists),np.max(test_mover_dists))

                            ds9_comm = 'ds9'
                            for k in range(len(movers[-1])):
                                ind = movers[-1][k]
                                print('   ',trans.images[ind],trans.x[ind],trans.y[ind],trans.snr[ind])
                                ds9_comm += ' */HSC-R2/corr/'+trans.images[ind].split('.')[0]+'.fits'
                                #ds9_comm += ' */'+trans.images[ind].split('.')[0]+'.fits'
                                ds9_comm += ' -pan to {} {} image'.format(trans.x[ind],trans.y[ind])
                                ds9_comm += ' -regions command "circle {} {} 10"'.format(trans.x[ind],trans.y[ind])

                                mover_details[-1].append([ind,trans.images[ind],trans.x[ind],trans.y[ind],trans.snr[ind]])
                            if showDS9string:
                                print(ds9_comm)
                            print()
                        else:
                            used[test_mover_ind] = 0

    print('Writing to {}'.format(blinkfn))
    with open(blinkfn,'wb') as han:
        pickle.dump(mover_details,han)
    print("we have found {} candidates.".format(len(mover_details)))
