import numpy as np
import pickle
from paths import *
from catObj import *
import time
from numba import jit

d2r = np.pi/180.0
r2d = 180.0/np.pi


@jit(target='cpu',nopython=True)
def gc_dist_d2r(x1,y1,x2,y2):
    a = np.sin((x2-x1)/2.0) ** 2.0 + (np.cos(x1) * np.cos(x2) * (np.sin((y2-y1)/2.0) ** 2.0))
    # Great circle distance in radians
    angle2 = 2.0 * np.arcsin(np.minimum(1.0, a**0.5))
    return angle2*r2d

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

    ra_d2r = brick.ra*d2r
    dec_d2r = brick.dec*d2r

    w = np.where(brick.snr<snr_min)
    wf[w] = False
    ws[w] = False
    wstat[w] = False

    n_det = len(brick.ra)
    print('Number of detections to consider: {}'.format(n_det))
    for ii in range(n_det):
        #skip those already marked as stationary
        if not wf[ii]: continue


        #fast distance estimate in ra/dec
        w_rd = np.where( (np.abs(brick.ra-brick.ra[ii])<0.0003/np.cos(dec_d2r[ii]))\
                        & (np.abs(brick.dec-brick.dec[ii])<0.0003)\
                        & (brick.snr>snr_min))

        #trim out the source in question
        j = np.where(w_rd[0]==ii)
        w_rd = [np.delete(w_rd[0],j[0][0])]

        #any sources within 0.0003 deg in both ra and dec and have a different time and position than the source in question?
        if len(w_rd[0])>0:

            w_t = np.where(np.abs(brick.jd[w_rd]-brick.jd[ii])>slow_time)

            #any sources with nearby sources on a different night?
            if len(w_t[0])>0:
                w = w_rd[0][w_t[0]]
                #dist = gc_dist_d2rdone(rad2r[ii],decd2r[ii],rad2r[w],decd2r[w])*3600.0
                dist = gc_dist_d2r(ra_d2r[ii],dec_d2r[ii],ra_d2r[w],dec_d2r[w])*3600.0

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
                        dist = gc_dist_d2r(ra_d2r[ii],dec_d2r[ii],ra_d2r[w],dec_d2r[w])*3600.0
                        #dist = gc_dist_d2rdone(rad2r[ii],decd2r[ii],rad2r[w],decd2r[w])*3600.0

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

    skips = [
'/media/fraserw/rocketdata/DEC2018/bricks/100114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/100115.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/100116.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/101114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/101115.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/101125.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/102125.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/103124.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/103125.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/104124.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/39095.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/39096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/39097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/40095.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/40096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/40097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/40098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/41093.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/41094.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/41095.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/41096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/41097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/41098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/41099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/42092.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/42093.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/42094.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/42095.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/42096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/42097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/42098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/42099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/43092.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/43093.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/43094.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/43095.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/43096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/43097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/43098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/43099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/43100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/44093.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/44094.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/44095.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/44096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/44097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/44098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/44099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/44100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/44101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/45093.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/45094.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/45095.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/45096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/45097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/45098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/45099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/45100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/45101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/46094.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/46095.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/46096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/46097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/46098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/46099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/46100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/46101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/46102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/47095.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/47096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/47097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/47098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/47099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/47100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/47101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/47102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/48095.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/48096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/48097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/48098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/48099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/48100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/48101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/48102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/48103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/49096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/49097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/49098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/49099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/49100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/49101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/49102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/49103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/49104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/50096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/50097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/50098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/50099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/50100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/50101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/50102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/50103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/50104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/51097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/51098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/51099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/51100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/51101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/51102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/51103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/51104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/51105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/52098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/52099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/52100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/52101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/52102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/52103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/52104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/52105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/52106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/53098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/53099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/53100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/53101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/53102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/53103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/53104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/53105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/53106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/53107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/53108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/54099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/54100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/54101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/54102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/54103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/54104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/54105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/54106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/54107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/54108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/55099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/55100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/55101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/55102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/55103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/55104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/55105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/55106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/55107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/56098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/56099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/56100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/56101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/56102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/56103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/56104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/56105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/56106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/56107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/56108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/57098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/57099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/57100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/57101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/57102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/57103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/57104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/57105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/57106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/57107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/57108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/57109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/58109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/59109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60095.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/60110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61095.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/61111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/62111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63096.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/63112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/64113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/65113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/66114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67097.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/67115.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/68115.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69098.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69099.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/69114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70100.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/70114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/71113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/72101.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/72102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/72103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/72104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/72105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/72106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/72107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/72108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/72109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/72110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/72111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/72112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/73102.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/73103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/73104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/73105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/73106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/73107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/73108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/73109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/73110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/73111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/73112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/74103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/74104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/74105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/74106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/74107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/74108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/74109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/74110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/74111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/74112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/74113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/75103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/75104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/75105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/75106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/75107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/75108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/75109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/75110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/75111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/75112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/75113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/76104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/76105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/76106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/76107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/76108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/76109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/76110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/76111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/76112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/76113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/76114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/77115.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78107.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78115.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79106.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79108.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79116.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/80103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/80104.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79115.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/80109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/78111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79103.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/80112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/81109.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/80111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/80116.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/80110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/80115.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/81112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/81114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/83111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/83116.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/84113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/84118.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/81113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/82110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/82115.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/83112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/83117.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/81114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/82111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/82116.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/83113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/83118.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/79114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/80105.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/80113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/81110.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/81115.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/82112.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/82117.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/80114.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/81111.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/81116.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/82113.brick',
'/media/fraserw/rocketdata/DEC2018/bricks/83110.brick',
    ]
    dos = [
    ]

    match_radius = 0.5 #arcseconds
    slow_time = 12.0/24.0 #12 hours into JD ,_ no moving object is stationary for 12 hours!
    stat_radius = 0.2 #arcseconds

    min_det_for_stat = 3

    snr_min = 5.0

    brick_files = glob.glob(masterDir+'/bricks/*brick')
    brick_files.sort()


    start = 0
    step = 1
    if len(sys.argv)>1:
        start = int(float(sys.argv[1]))
        step = int(float(sys.argv[2]))

    for i in range(start,len(brick_files),step):
        if brick_files[i].split('/')[-1] not in dos and len(dos)>0:
            continue
        if brick_files[i] in skips:
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
