import multiprocessing as multip
import os,glob,sys
from paths import *

def runFinder(bn_i,diff=''):
    print('Running {}'.format(bn_i))
    os.system('/home/fraserw/anaconda3/bin/python p9findmoving.py {} {}'.format(bn_i,diff))


test_single = False
if test_single:
    bn_i = 2
    runFinder(bn_i)

else:

    if len(sys.argv)>1:
        if '--diff' not in sys.argv:
            print('Pass --diff to run on the difference catalog.')
            exit()
        masterDir += '/DiffCatalog'

    brick_files = glob.glob(masterDir+'/bricks/*brick')
    brick_files.sort()

    blink_files = glob.glob(masterDir+'/veryfast_blinks/blink_veryfast.*')
    skips = []
    for i in range(len(brick_files)):
        skip = False
        bn = brick_files[i].split('/')[-1].split('.')[0]
        for j in range(len(blink_files)):
            if bn in blink_files[j]:
                skip = True
        if skip:
            skips.append(i)
        else:
            print(i,brick_files[i])

    nbf = len(brick_files)



    numJobs = 6
    q = multip.Queue()
    i = 0
    while i < nbf+1:
        processes = []
        for j in range(numJobs):
            if i+j ==  nbf: break
            if i+j in skips: continue
            bn_i = i+j
            if '--diff' in sys.argv:
                processes.append(multip.Process(target=runFinder, args=(bn_i,'--diff')))
            else:
                processes.append(multip.Process(target=runFinder, args=(bn_i,'--veryfast')))
            processes[-1].start()
        for j in range(len(processes)):
            processes[j].join()
        i += numJobs
