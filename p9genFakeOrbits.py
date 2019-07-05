import ephem as eph
import numpy as np,pylab as pyl, scipy as sci

d2r = np.pi/180.0
r2d = 180.0/np.pi


with open('/media/fraserw/rocketdata/DEC2018/corners.txt') as han:
    data = han.readlines()
rs = []
ds = []
mjds = []
for i in range(len(data)):
    s = data[i].split()
    mjds.append(float(s[1]))
    rs.append(float(s[2]))
    rs.append(float(s[4]))
    rs.append(float(s[6]))
    rs.append(float(s[8]))
    ds.append(float(s[3]))
    ds.append(float(s[5]))
    ds.append(float(s[7]))
    ds.append(float(s[9]))
mjds = np.array(mjds)
rs = np.array(rs)
ds = np.array(ds)

min_mjd = np.min(mjds)
max_mjd = np.max(mjds)
min_ra = np.min(rs)
max_ra = np.max(rs)
min_dec = np.min(ds)
max_dec = np.max(ds)

print(min_ra,max_ra)
print(min_dec,max_dec)


jds = np.array([min_mjd,max_mjd])+2400000.5
djds = jds - 2415020

observer = eph.Observer()
observer.lon = '204:31:40.1'
observer.lat = '19:49:34.0'
observer.elevation = 4212


kbos = []
outhan = open('kbos_to_plant.dat','w+')
while len(kbos)<500000:

    kbo = eph.EllipticalBody()
    (a,e,inc,a0,a1,a2,djdM,djd,m) = sci.rand(9)*np.array([950.0,0.6,90.0,360.0,360.0,360.0,2000.0,2000.0,10]) + np.array([50.0,0.0,0.0,0.0,0.0,0.0,djds[0],djds[0],17])
    kbo._a = a
    kbo._e = e
    kbo._inc = inc
    kbo._M = a0
    kbo._Om = a1
    kbo._om = a2
    kbo._epoch_M = djdM
    kbo._epoch = djd



    ra = []
    dec = []
    sd = []
    ed = []
    for i in range(2):
        observer.date = djds[i]
        kbo.compute(observer)
        ra.append(float(kbo.a_ra)*r2d)
        dec.append(float(kbo.a_dec)*r2d)
    H = m-5.0*np.log10(kbo.sun_distance*kbo.earth_distance)

    #print(ra,dec)
    if (ra[0]>=min_ra and ra[0]<=max_ra and dec[0]>min_dec and dec[0]<=max_dec) or (ra[1]>=min_ra and ra[1]<=max_ra and dec[1]>min_dec and dec[1]<=max_dec):
        kbos.append([a,e,inc,a0,a1,a2,djdM,djd,H])
        print(len(kbos),ra[0],dec[0],kbos[-1][0],kbos[-1][1],kbos[-1][2],kbos[-1][3],kbos[-1][4],kbos[-1][5],kbos[-1][6],kbos[-1][7],kbos[-1][8],m,file = outhan)
outhan.close()
