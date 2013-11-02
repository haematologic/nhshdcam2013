from scipy.misc import imread, imsave
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction import image
from IPython import embed

images = [ 'images/%s' % f for f in listdir('images') ]

c = 0
 
for img in images:
    data = imread(img)
    patches = image.extract_patches_2d(data, (64, 64), max_patches=512)
    for p in xrange(0, patches.shape[0]):
        val = (patches[p,:,:,2] - patches[p,:,:].mean()).mean()
        if val > 10:
            imsave('data/%d-%d-%d.jpg' % (c,p, int(val * 100)), patches[p,:,:])
    c = c + 1
    print "Processing: %d" % c
