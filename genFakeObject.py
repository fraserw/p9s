import ephem as eph
import scipy as sci, numpy as np

r2d = 180.0/np.pi


a = 300.0
e = 0.3
inc = 30.0


def getradec(jds,a=500.0,e=0.3,inc=30.0,niter=1000000):
    djd = jds[1] - 2415020

    observer = eph.Observer()
    observer.date = djd

    observerm1 = eph.Observer()
    observerm1.date = jds[0]-2415020

    observerp1 = eph.Observer()
    observerp1.date = jds[2]-2415020

    ras = []
    decs = []
    n = -1
    while n<niter:
        body = eph.EllipticalBody()
        body._inc = inc
        body._a = a
        body._e = e
        ang = sci.rand(3)*360.0
        body._om = ang[0]
        body._Om = ang[1]
        body._M = ang[2]
        body._epoch = djd
        body._epoch_M = djd

        body.compute(observer)
        ras.append(body.a_ra*r2d)
        decs.append(body.a_dec*r2d)

        if abs(ras[-1]-43.2)<1 and abs(decs[-1]-2)<1:
            body.compute(observerm1)
            ra_m1 = body.a_ra*r2d
            dec_m1 = body.a_dec*r2d
            body.compute(observerp1)
            ra_p1 = body.a_ra*r2d
            dec_p1 = body.a_dec*r2d
            #print(ra_p1,ra_m1,dec_p1,dec_m1)

            print(ras[-1],decs[-1],body.sun_distance)
            print((ra_p1-ras[-1])*3600.0/24,(dec_p1-decs[-1])*3600.0/24)
            print((ras[-1]-ra_m1)*3600.0/24,(decs[-1]-dec_m1)*3600.0/24)
            print()
            return (np.array([ra_m1,ras[-1],ra_p1]),np.array([dec_m1,decs[-1],dec_p1]))
        n+=1
if __name__ == "__main__":
    jds = [2458018.9419906866, 2458019.9409898967, 2458020.9314896464]
    getradec(jds)
