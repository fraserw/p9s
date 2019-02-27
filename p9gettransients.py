import numpy as np
import pickle

from paths import *
from catObj import *

d2r = np.pi/180.0
r2d = 180.0/np.pi

def gc_dist(x1,y1,x2,y2):
    X1 = x1*d2r
    X2 = x2*d2r
    Y1 = y1*d2r
    Y2 = y2*d2r
    a = np.sin((X2-X1)/2.0) ** 2.0 + (np.cos(X1) * np.cos(X2) * (np.sin((Y2-Y1)/2.0) ** 2.0))
    # Great circle distance in radians
    angle2 = 2.0 * np.arcsin(np.minimum(1.0, a**0.5))
    return angle2*r2d


def searchBrick(brick,match_radius=0.75,slow_time=0.5,stat_radius=0.2,min_det_for_stat=3,snr_min=5.0):
    wf = np.full(brick.ra.shape,True,dtype='bool')
    ws = np.full(brick.ra.shape,False,dtype='bool')
    wstat = np.full(brick.ra.shape,False,dtype='bool')

    n_det = len(brick.ra)
    print('Number of detections to consider: {}'.format(n_det))
    for ii in range(n_det):
        #skip those already marked as stationary
        if not wf[ii]: continue

        if brick.snr[ii]<snr_min:
            wf[ii] = False
            ws[ii] = False
            wstat[ii] = False
            continue

        #fast distance estimate in ra/dec
        w_rd = np.where( (np.abs(brick.ra-brick.ra[ii])<0.0003/np.cos(brick.dec[ii]*d2r)) & (np.abs(brick.dec-brick.dec[ii])<0.0003) & (brick.snr>snr_min) & (brick.ra[ii]!=brick.ra) & (brick.dec[ii]!=brick.dec) & (brick.jd[ii]!=brick.jd))

        #any sources within 0.0003 deg in both ra and dec and have a different time and position than the source in question?
        if len(w_rd[0])>0:

            w_t = np.where(np.abs(brick.jd[w_rd]-brick.jd[ii])>slow_time)

            #any sources with nearby sources on a different night?
            if len(w_t[0])>0:
                w = w_rd[0][w_t[0]]
                dist = gc_dist(brick.ra[ii],brick.dec[ii],brick.ra[w],brick.dec[w])*3600.0

                w1 = np.where(dist<match_radius)
                if len(w1[0])>0:
                    #dulicated on a separate night, not transient
                    wf[ii] = False
                    wf[w[w1]] = False

                    w2 = np.where(dist<stat_radius)
                    if len(w2[0])>=min_det_for_stat:
                        wstat[ii] = True
                        wstat[w[w2]] = True

                else:
                    #not matched as stationary. This must be a transient!

                    #I THINK THERE IS A BUG HERE. That's because w_rd only counts source clumps within ~1.1" of the source in question, but
                    #an object at 100 AU will move ~1.5"/hr at opposition. So I think there is potential to miss those distant objects
                    #if the imaging revisit time is more than 45 minutes away.
                    dt = np.abs(brick.jd[w_rd] - brick.jd[ii])
                    w_t_slow = np.where( (dt<slow_time) & (dt>0.0007)) #1 minute, just to make sure this isn't exactly zero.

                    if len(w_t_slow[0])>0:
                        #we have at least one other detection more than a minute apart
                        #that also hasn't been identified as a local source from another night
                        #that also has a nearby source on a different night
                        w = w_rd[0][w_t_slow[0]]
                        dist = gc_dist(brick.ra[ii],brick.dec[ii],brick.ra[w],brick.dec[w])*3600.0
                        w1 = np.where(dist<match_radius)
                        if len(w1[0])>0:
                            #we have more than one detection on the same day
                            #seems liks a slow moving transient
                            ws[ii] = True
                            ws[w[w1]] = True
    brick.ws = np.copy(ws)
    brick.wf = np.copy(wf)
    brick.wstat = np.copy(wstat)

    return brick



if __name__ == "__main__":
    import glob,sys

    match_radius = 0.75 #arcseconds
    slow_time = 12.0/24.0 #12 hours into JD ,_ no moving object is stationary for 12 hours!
    stat_radius = 0.2 #arcseconds

    min_det_for_stat = 3

    snr_min = 5.0

    brick_files = glob.glob(masterDir+'/bricks/*brick')


    dos = [

    ]

    start = 0
    step = 1
    if len(sys.argv)>1:
        start = int(float(sys.argv[1]))
        step = int(float(sys.argv[2]))

    for i in range(start,len(brick_files),step):
        if brick_files[i].split('/')[-1] not in dos and len(dos)>0:
            continue

        print(brick_files[i],i+1,len(brick_files))

        with open(brick_files[i],'rb') as han:
            brick = pickle.load(han)


        brick = searchBrick(brick,match_radius,slow_time,stat_radius,min_det_for_stat,snr_min)

        unassigned = np.where( (brick.wstat == False) & (brick.ws == False) & (brick.wf == False))
        #print('Number of unassigned sources is',len(unassigned[0]))
        print('Number of slow transients is',len(np.where(brick.ws==True)[0]))
        pickle.dump(brick,open(brick_files[i],'wb'))
        print('')
