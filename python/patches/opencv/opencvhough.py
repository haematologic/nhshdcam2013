from IPython import embed
import cv2
import numpy as np
import time
import sys
from matplotlib import pyplot as plt
from scipy.misc import imread, imsave
from os import listdir
from os.path import isfile, join

images = [ '../images/%s' % f for f in listdir('../images') ]
c = 0

for image in images:
    c = c+1

    if len(sys.argv)>1:
        filename = sys.argv[1]
    else:
        filename = image

    img = cv2.imread(filename,0)
    imgcol = cv2.imread(filename,1)
    imgcol = cv2.cvtColor(imgcol,cv2.COLOR_BGR2RGB)	
    if img==None:
        print "cannot open ",filename

    else:
        img = cv2.medianBlur(img,3)
        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # Python: cv2.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2 18 [, minRadius 10 [, maxRadius]]]]])
        circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,15,param1=100,param2=18,minRadius=10,maxRadius=40)
        #embed()
	circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1)  # draw the outer circle
	    pad = 5
	    size = 32
	    # size = i[2]
	    xstart = i[0]-size-pad
	    xfinish = i[0]+size+pad
	    ystart = i[1]-size-pad
	    yfinish = i[1]+size+pad
	    if xstart < 0:
		xstart = 0
		xfinish = size*2
	    if ystart < 0:
		ystart = 0
		yfinish = size*2
 	    if xfinish > 1024:
		xfinish = 1024
		xstart = 1024-(size*2)
	    if yfinish > 1024:
		yfinish = 1024
		ystart = 1024-(size*2)
            cropImg = imgcol[ystart:yfinish, xstart:xfinish]
	    #plt.imshow(cropImg)
	    #plt.show()
	    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)     # draw the center of the circle
	    val = (cropImg[:,:,2] - cropImg[:,:].mean()).mean()
            #embed()
	    print(filename, 'i= ',i[0],i[1],i[2], 'val=',val) 
	    if abs(val) > 10.0:
    	        imsave('output/%d-%d-%d-%d.jpg' % (c,xstart,ystart,int(val*100)),cropImg)
        #cv2.imshow('detected circles',cimg)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
