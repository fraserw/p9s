import numpy as np

masterDir = '/media/fraserw/Hammer/SEP2017'
sourceDir = masterDir+'/02091/'  #working with data from /net/eris/data1/surhud/P9_2017_frames/02091 at the moment
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
