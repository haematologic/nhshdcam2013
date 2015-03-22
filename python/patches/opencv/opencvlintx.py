''' Simple and fast image transforms to mimic:
 - brightness
 - contrast
 - erosion
 - dilation '''

import matplotlib
matplotlib.use('TkAgg') 
import cv2 
from pylab import array, plot, show, axis, arange, figure, uint8 
from os import listdir 
from os.path import isfile, join

images = [ 'output/channels/%s' % f for f in listdir('output/channels') ]

c = 0

for image in images:
    #embed()
    c = c+1

    # Image data
    image = cv2.imread(image,0) # load as 1-channel 8bit grayscale
    cv2.imshow('image',image)
    cv2.waitKey(0) == ord('a') 
    maxIntensity = 255.0 # depends on dtype of image data
    x = arange(maxIntensity)

    # Parameters for manipulating image data
    phi = 1
    theta = 1

    # Increase intensity such that dark pixels become much brighter, 
    # bright pixels become slightly bright
    newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
    newImage0 = array(newImage0,dtype=uint8)

    cv2.imshow('%d-newImage0' % (c), newImage0)
    cv2.waitKey(0) == ord('a')
    cv2.imwrite('%d-newImage0.jpg' % (c), newImage0)

    y = (maxIntensity/phi)*(x/(maxIntensity/theta))**0.5

    # Decrease intensity such that dark pixels become much darker, 
    # bright pixels become slightly dark
    newImage1 = (maxIntensity/phi)*(image/(maxIntensity/theta))**2
    newImage1 = array(newImage1,dtype=uint8)

    cv2.imshow('%d-newImage1' % (c), newImage1)
    cv2.waitKey(0) == ord('a')
    z = (maxIntensity/phi)*(x/(maxIntensity/theta))**2

    # Plot the figures
    figure()
    plot(x,y,'r-') # Increased brightness
    plot(x,x,'k:') # Original image
    plot(x,z, 'b-') # Decreased brightness
    #axis('off')
    axis('tight')
    show()

    # Close figure window and click on other window Then press any 
    # keyboard key to close all windows
    closeWindow = -1
    while closeWindow<0:
        closeWindow = cv2.waitKey(1)
    cv2.destroyAllWindows()

