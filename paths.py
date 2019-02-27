import numpy as np

#ran extract on 02091,02093
masterDir = '/media/fraserw/Thumber/DEC2018_also'
sourceDir = masterDir+'/02529/HSC-R2/corr/'  #working with data from /net/eris/data1/surhud/P9_2017_frames/02091 at the moment
masksDir = '/home/fraserw/idl_progs/hscp9/sextract'

cutRegion = [0,4176,0,2048]

sepPixStack = 600000

maskChips = [
'000',
'004',
'006',
'009',
'022',
'029',
'030',
'033',
'037',
'043',
'045',
'054',
'062',
'069',
'070',
'074',
'077',
'090',
'095',
'100',
'101',
'102',
'103']

d2r = np.pi/180.0
r2d = 180.0/np.pi

apertures = {2:0,3:0,4:0,5:0,6:1,7:1,8:2,9:2,10:3,11:3,12:4,13:4,14:4,15:4,16:4,17:4,18:4,19:4,20:4}
