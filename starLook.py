import scamp,numpy as np, pylab as pyl
from p9extract import getMeanMagDiff
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import STU
import SpanningTrees as ST
from scipy import cluster
from astropy.io import fits
from astropy.visualization import interval
from stsci import numdisplay
from sklearn import cluster,mixture

def skcluster(sample):
    gmm = mixture.GaussianMixture(
        n_components=3, covariance_type='full')
    gmm.fit(sample)
    pred = gmm.predict(sample)
    print(pred)
    #spectral = cluster.SpectralClustering(
    #    n_clusters=3, eigen_solver='arpack',
    #    affinity="nearest_neighbors")
    #spectral.fit(sample)
    #pred = spectral.labels_.astype(np.int)
    return pred

def getTree(sample):
    nodes = []
    for i in range(len(sample)):
        nodes.append(str(i))

    edges = []
    for i in range(len(sample)):
        for j in range(len(sample)):
            if i!=j:
                d = ((sample[i][0]-sample[j][0])**2 + (sample[i][1]-sample[j][1])**2 + (sample[i][2]-sample[j][2])**2)**0.5
                edges.append((str(i),str(j),d))

    tree = ST.prim(nodes,edges)
    tree = sorted(tree, key = lambda a : a[2], reverse = True)
    return tree

def getSubTrees(sample,result):

    ids_0 = []
    for i in range(len(result[7])):
        i0 = int(tree[result[7][i]][0])
        i1 = int(tree[result[7][i]][1])
        ids_0.append(i0)
        ids_0.append(i1)
    ids_1 = []
    for i in range(len(result[8])):
        i0 = int(tree[result[8][i]][0])
        i1 = int(tree[result[8][i]][1])
        ids_1.append(i0)
        ids_1.append(i1)
    unique_ids_0 = np.unique(np.array(ids_0))
    unique_ids_1 = np.unique(np.array(ids_1))

    sub_sample_0 = sample[unique_ids_0]
    sub_sample_1 = sample[unique_ids_1]

    return (sub_sample_0,getTree(sub_sample_0),sub_sample_1,getTree(sub_sample_1))


def plotTree(resu,samp,stree,sp,skclass = True,pred=[]):

    for j in range(len(resu[7])):
        i0 = int(stree[resu[7][j]][0])
        i1 = int(stree[resu[7][j]][1])
        if resu[7][j]!=resu[2]:
            sp.plot([samp[i0][0],samp[i1][0]],[samp[i0][1],samp[i1][1]],[samp[i0][2],samp[i1][2]],'r-',lw=1.5)
    for j in range(len(resu[8])):
        i0 = int(stree[resu[8][j]][0])
        i1 = int(stree[resu[8][j]][1])
        if resu[8][j]!=resu[2]:
            sp.plot([samp[i0][0],samp[i1][0]],[samp[i0][1],samp[i1][1]],[samp[i0][2],samp[i1][2]],'b-',lw=1.5)

    i0 = int(stree[resu[2]][0])
    i1 = int(stree[resu[2]][1])
    sp.plot([samp[i0][0],samp[i1][0]],[samp[i0][1],samp[i1][1]],[samp[i0][2],samp[i1][2]],'k-.',lw=1.5)
    if not skclass:
        sp.scatter(samp[:,0],samp[:,1],samp[:,2],marker='o',color='k')
    else:
        cs = ['k','r','b','g']
        colours = np.full(len(pred),cs[0],dtype = 'str')
        u = np.unique(pred)
        for jj in range(len(u)):
            w = np.where(pred==u[jj])
            colours[w]=cs[jj]

        sp.scatter(samp[:,0],samp[:,1],samp[:,2],marker='o',color=colours)
    return None

def plotCutoutsTree(resu,stree,pos_x,pos_y,fits_fn,ind=7,cutSize = 25,show = True):
    with fits.open(fits_fn) as han:
        data = han[1].data
    ui = []
    for j in range(len(resu[ind])):
        i0 = int(stree[resu[ind][j]][0])
        i1 = int(stree[resu[ind][j]][1])
        ui.append(i0)
        ui.append(i1)
    ui = np.unique(np.array(ui))
    nsp = int(len(ui)**0.5)
    if nsp*nsp<len(ui):
        nsp += 1


    fig = pyl.figure('cutouts_'+str(ind))
    gs = gridspec.GridSpec(nsp,nsp)

    j = 0
    k = 0
    npl = 0
    for ii in range(len(ui)):
        x = int(pos_x[ui[ii]])
        y = int(pos_y[ui[ii]])
        cut = data[y-cutSize:y+cutSize+1,x-cutSize:x+cutSize+1]

        (z1,z2) = numdisplay.zscale.zscale(cut,contrast = 0.5)
        normer = interval.ManualInterval(z1,z2)
        Cut = normer(cut)

        sp = pyl.subplot(gs[j,k])
        sp.imshow(Cut)
        j+=1
        npl +=1
        if j==nsp:
            j = 0
            k+=1
        if npl == nsp*nsp:
            break
    if show:
        pyl.show()


def plotCutoutsClass(pred,pos_x,pos_y,fits_fn,ind=0,cutSize = 25,show = True):
    with fits.open(fits_fn) as han:
        data = han[1].data
    ui = np.arange(len(pos_x))
    w = np.where(pred == ind)
    ui = ui[w]

    nsp = int(len(ui)**0.5)
    if nsp*nsp<len(ui):
        nsp += 1

    nsp = min(nsp,10)
    print(nsp,len(ui))


    fig = pyl.figure('cutouts_'+str(ind))
    gs = gridspec.GridSpec(nsp,nsp)

    j = 0
    k = 0
    npl = 0
    for ii in range(len(ui)):
        x = int(pos_x[ui[ii]])
        y = int(pos_y[ui[ii]])
        cut = data[y-cutSize:y+cutSize+1,x-cutSize:x+cutSize+1]

        (z1,z2) = numdisplay.zscale.zscale(cut,contrast = 0.5)
        normer = interval.ManualInterval(z1,z2)
        Cut = normer(cut)

        sp = pyl.subplot(gs[j,k])
        sp.imshow(Cut)
        j+=1
        npl+=1
        if j==nsp:
            j = 0
            k+=1
        if npl == nsp*nsp:
            break
    if show:
        pyl.show()


header = {'EXPTIME':200.0,'MAGZERO':26.0}

apertures = {2:0,3:0,4:0,5:0,6:1,7:1,8:2,9:2,10:3,11:3,12:4,13:4,14:4,15:4,16:4,17:4,18:4,19:4,20:4}
apNum = 4

#shitty seeing image
fn = '/media/fraserw/Hammer/DEC2018/02531/HSC-R2/corr/sexSaves/CORR-0154776-084.cat'
fits_fn = '/media/fraserw/Hammer/DEC2018/02531/HSC-R2/corr/CORR-0154776-084.fits'

#good seeing image
fn = '/media/fraserw/rocketdata/SEP2017/02093/sexSaves/CORR-0132546-034.cat'
fits_fn = 'CORR-0132546-034.fits'

catalog = scamp.getCatalog(fn,paramFile='sextract.param')





w = np.where((catalog['X_IMAGE']>50) & (catalog['X_IMAGE']<1995) & (catalog['Y_IMAGE']>50) & (catalog['Y_IMAGE']<4123) \
    &  ((catalog['FLUX_APER(5)'][:,apNum]/catalog['FLUXERR_APER(5)'][:,apNum])>50) & (catalog['FWHM_IMAGE']>1.5))

for i in catalog:
    #print(i)
    catalog[i] = catalog[i][w]

moment = catalog['X2_IMAGE']/catalog['Y2_IMAGE']
#w = np.where(moment<1)
#moment[w] = 1.0/moment[w]


pos_x = catalog['X_IMAGE']
pos_y = catalog['Y_IMAGE']

mag_aper = -2.5*np.log10(catalog['FLUX_APER(5)'][:,apNum]/header['EXPTIME'])+header['MAGZERO']
mag_auto = -2.5*np.log10(catalog['FLUX_AUTO']/header['EXPTIME'])+header['MAGZERO']

mag_diff = mag_auto - mag_aper
med_mag_diff = getMeanMagDiff(mag_aper,mag_diff)
mag_diff -= med_mag_diff

snr = catalog['FLUX_APER(5)'][:,apNum]/catalog['FLUXERR_APER(5)'][:,apNum]

x = catalog['FWHM_IMAGE']
y = mag_diff
z = catalog['A_IMAGE']/catalog['B_IMAGE']
xx = moment


sample  = np.zeros((len(x),4)).astype('float64')
sample[:,0] = xx
sample[:,1] = y
sample[:,2] = z
sample[:,3] = x


tree = getTree(sample)
result = STU.fraserStat(tree,sample,thetaStep=5.)


(sub_sample_0,sub_tree_0,sub_sample_1,sub_tree_1) = getSubTrees(sample,result)

sub_result_0 = STU.fraserStat(sub_tree_0, sub_sample_0, thetaStep=5.)
sub_result_1 = STU.fraserStat(sub_tree_1, sub_sample_1, thetaStep=5.)

#plotCutouts(sub_result_0,sub_tree_0,pos_x,pos_y,fits_fn,ind=7,show = False)
#plotCutouts(sub_result_0,sub_tree_0,pos_x,pos_y,fits_fn,ind=8,show =True)
#exit()



#(dist,clas) = (cluster.vq.kmeans2(sample,3))
pred = skcluster(sample)
lp = [100000,-1]
for p in np.unique(pred):
    w = np.where(pred==p)
    m = np.mean(sample[:,3][w])
    if m < lp[0]:
        lp = [m,p]
with open(fits_fn.replace('.fits','.psfStars'),'w+') as han:
    w = np.where(pred == lp[1])
    for i in range(len(w[0])):
        han.write('{:>8.2f} {:>8.2f} {:>10.2f} {:>10.2f}\n'.format(catalog['X_IMAGE'][w[0][i]],catalog['Y_IMAGE'][w[0][i]],catalog['FLUXERR_APER(5)'][:,apNum][w[0][i]],catalog['FLUX_APER(5)'][:,apNum][w[0][i]]))

exit()
plotCutoutsClass(pred,pos_x,pos_y,fits_fn,ind=2,cutSize = 25,show = False)
plotCutoutsClass(pred,pos_x,pos_y,fits_fn,ind=1,cutSize = 25,show = False)
plotCutoutsClass(pred,pos_x,pos_y,fits_fn,ind=0,cutSize = 25,show = False)
fig = pyl.figure()
sp = fig.add_subplot(111, projection='3d')

#plotTree(sub_result_0,sub_sample_0,sub_tree_0,sp)
plotTree(result, sample, tree,sp,skclass = True, pred=pred)

sp.set_xlabel('FWHM_IMAGE')
sp.set_ylabel('Aper - Kron')
sp.set_zlabel('A/B')

pyl.show()
exit()



for j in range(len(result[7])):
    i0 = int(tree[result[7][j]][0])
    i1 = int(tree[result[7][j]][1])
    if result[7][j]!=result[2]:
        sp.plot([sample[i0][0],sample[i1][0]],[sample[i0][1],sample[i1][1]],[sample[i0][2],sample[i1][2]],'r-',lw=1.5)
for j in range(len(result[8])):
    i0 = int(tree[result[8][j]][0])
    i1 = int(tree[result[8][j]][1])
    if result[8][j]!=result[2]:
        sp.plot([sample[i0][0],sample[i1][0]],[sample[i0][1],sample[i1][1]],[sample[i0][2],sample[i1][2]],'b-',lw=1.5)

i0 = int(tree[result[2]][0])
i1 = int(tree[result[2]][1])
sp.plot([sample[i0][0],sample[i1][0]],[sample[i0][1],sample[i1][1]],[sample[i0][2],sample[i1][2]],'k-.',lw=1.5)




colors = []
for i in range(len(x)):
    colors.append('b')
colors = np.array(colors)
w = np.where(clas==1)
colors[w] = 'r'
w = np.where(clas==2)
colors[w] = 'y'

sp.scatter(sample[:,0],sample[:,1],sample[:,2],marker='o',color = colors)
sp.set_ylabel('Aper - Kron')
sp.set_xlabel('FWHM_IMAGE')
sp.set_zlabel('A/B')

#sp.set_xlim(0,10)
pyl.show()
