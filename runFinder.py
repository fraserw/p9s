import multiprocessing as multip
import os,glob,sys
from paths import *

def runFinder(bn_i,diff=''):
    print('Running {}'.format(bn_i))
    #os.system('/home/fraserw/anaconda3/bin/python p9findmoving.py {} {} > junk_output/{}.junk'.format(bn_i,diff,bn_i))
    os.system('/home/fraserw/anaconda3/bin/python p9findmoving.py {} {}'.format(bn_i,diff))

#masterDir = '/media/fraserw/rocketdata/CTIO_DEC_2018'

test_single = False
if test_single:
    brick_files = glob.glob(masterDir+'/bricks/*brick')
    brick_files.sort()

    bn_i = 29

    runFinder(bn_i,'--fast')

else:

    if len(sys.argv)>1:
        if '--diff' not in sys.argv and '--fast' not in sys.argv and '--veryfast' not in sys.argv and '--ctio' not in sys.argv:
            print('Pass --diff to run on the difference catalog.')
            exit()
        if '--diff' in sys.argv: masterDir += '/DiffCatalog'

    brick_files = glob.glob(masterDir+'/bricks/*brick')
    brick_files.sort()


    if '--fast' in sys.argv:
        brick_files = glob.glob(masterDir+'/bricks/*brick')
        blink_files = glob.glob(masterDir+'/fast_blinks/blink_fast.*')
    elif '--veryfast' in sys.argv:
        brick_files = glob.glob(masterDir+'/bricks/*brick')
        blink_files = glob.glob(masterDir+'/veryfast_blinks/blink_veryfast.*')
    elif '--ctio' in sys.argv:
        brick_files = glob.glob(masterDir+'/bricks/*brick')
        blink_files = glob.glob(masterDir+'/blinks/blink.*')
    else:
        brick_files = glob.glob(masterDir+'/bricks/*brick')
        blink_files = glob.glob(masterDir+'/blinks/blink.*')

    brick_files.sort()
    blink_files.sort()

#caltech UID 1764425
    skips = []
    for i in range(len(brick_files)):
        skip = False
        bn = brick_files[i].split('/')[-1].split('.')[0]
        for j in range(len(blink_files)):
            if bn in blink_files[j]:
                skip = True
        #skip = False
        if skip:
            skips.append(bn)
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
            bn = brick_files[i+j].split('/')[-1].split('.')[0]
            if bn in  skips:
                print('Skipping',brick_files[i+j])
                continue
            bn_i = i+j
            if '--diff' in sys.argv:
                processes.append(multip.Process(target=runFinder, args=(bn_i,'--diff')))
            elif '--fast' in sys.argv:
                processes.append(multip.Process(target=runFinder, args=(bn_i,'--fast')))
            elif '--veryfast' in sys.argv:
                processes.append(multip.Process(target=runFinder, args=(bn_i,'--veryfast')))
            elif '--ctio' in sys.argv:
                processes.append(multip.Process(target=runFinder, args=(bn_i,'--ctio')))
            else:
                #processes.append(multip.Process(target=runFinder, args=(bn_i,'--veryfast')))
                processes.append(multip.Process(target=runFinder, args=(bn_i,)))
            processes[-1].start()
        for j in range(len(processes)):
            processes[j].join()
        i += numJobs
