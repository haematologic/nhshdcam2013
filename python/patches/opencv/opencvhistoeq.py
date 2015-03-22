from os import listdir
import cv2
import numpy as np
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
from IPython import embed
from scipy.misc import imread, imsave
images = [ '../images/%s' % f for f in listdir('../images') ]
c = 0

for image in images:
      
     c = c+1
     embed() 
     if len(sys.argv)>1:
         filename = sys.argv[1]
     else:
         filename = image
     img = cv2.imread(filename,1)

     hist,bins = np.histogram(img.flatten(),256,[0,256])

     cdf = hist.cumsum()
     cdf_normalized = cdf *hist.max()/ cdf.max() # line not necessary.

     plt.plot(cdf_normalized, color = 'b')
     plt.hist(img.flatten(),256,[0,256], color = 'r')
     plt.xlim([0,256])
     plt.legend(('cdf','histogram'), loc = 'upper left')
     plt.show()

     cdf_m = np.ma.masked_equal(cdf,0)
     cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
     cdf = np.ma.filled(cdf_m,0).astype('uint8')

     img2 = cdf[img]

     hist,bins = np.histogram(img2.flatten(),256,[0,256])

     cdf = hist.cumsum()
     cdf_normalized = cdf *hist.max()/ cdf.max()

     plt.plot(cdf_normalized, color = 'b')
     plt.hist(img2.flatten(),256,[0,256], color = 'r')
     plt.xlim([0,256])
     plt.legend(('cdf eq','histogram eq'), loc = 'upper left')
     plt.show()
     
     imsave('output/%s-%d.jpg' % ('img2',c),img2)
