from IPython import embed
import cv2
import numpy as np
import time
import sys

if len(sys.argv)>1:
    filename = sys.argv[1]
else:
    filename = '../images/test.jpg'

img = cv2.imread(filename,0)
if img==None:
    print "cannot open ",filename

else:
    img = cv2.medianBlur(img,3)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # Python: cv2.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]])
    circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,15,param1=100,param2=18,minRadius=10,maxRadius=35)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1)  # draw the outer circle
        print(i[2])
	cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)     # draw the center of the circle

    cv2.imshow('detected circles',cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
