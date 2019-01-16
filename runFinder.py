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
    nbf = len(brick_files)

    numJobs = 6
    q = multip.Queue()
    i = 0
    while i < nbf+1:
        processes = []
        for j in range(numJobs):
            if i+j ==  nbf: break

            bn_i = i+j
            if '--diff' in sys.argv:
                processes.append(multip.Process(target=runFinder, args=(bn_i,'--diff')))
            else:
                processes.append(multip.Process(target=runFinder, args=(bn_i,)))
            processes[-1].start()
        for j in range(len(processes)):
            processes[j].join()
        i += numJobs
