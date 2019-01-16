import numpy as np

class catObj(object):
    def __init__(self,catFile=None):
        self.x = None
        self.y = None
        self.ra = None
        self.dec = None
        self.flux = None
        self.snr = None
        self.mag = None
        self.fwhm_image = None
        self.jd = None
        self.seeing = None
        self.ellip = None
        self.astrms = None
        self.brick = None
        self.images = None

        if catFile is not None:

            self.x = []
            self.y = []
            self.ra = []
            self.dec = []
            self.flux = []
            self.snr = []
            self.mag = []
            self.fwhm_image = []
            self.jd = []
            self.seeing = []
            self.ellip = []
            self.astrms = []
            self.brick = []
            self.images = []

            with open(catFile) as han:
                catDat = han.readlines()
            for i in range(len(catDat)):
                s = catDat[i].split(',')
                try:
                    mag = float(s[0])
                except:
                    continue

                self.mag.append(mag)
                self.jd.append(float(s[1])+2400000.5)
                self.snr.append(float(s[4]))
                self.ra.append(float(s[5]))
                self.dec.append(float(s[6]))
                self.x.append(float(s[14]))
                self.y.append(float(s[15]))
                self.fwhm_image.append(float(s[12]))
                self.images.append('CORR-'+s[13].zfill(7)+'-'+s[16].zfill(3)+'.fits')

        self.mag = np.array(self.mag)
        self.snr = np.array(self.snr)
        self.jd = np.array(self.jd)
        self.ra = np.array(self.ra)
        self.dec = np.array(self.dec)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.fwhm_image = np.array(self.fwhm_image)
        self.images = np.array(self.images)




class brickObj(object):
    def __init__(self):
        self.x = []
        self.y = []
        self.ra = []
        self.dec = []
        self.flux = []
        self.snr = []
        self.mag = []
        self.fwhm_image = []
        self.jd = []
        self.seeing = []
        self.ellip = []
        self.astrms = []
        self.images = []
        self.brick = []
        self.wf = [] #potential fast moving transient
        self.ws = [] #potential slow moving transient
        self.wstat = [] #confirmed stationary source
        self.images = []
