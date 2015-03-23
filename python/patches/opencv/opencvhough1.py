from IPython import embed
import cv2
import numpy as np
np.set_printoptions(threshold='nan')
import time
import math
import sys
import matplotlib
matplotlib.use('TKAgg')
import mahotas as mh
from matplotlib import pyplot as plt
from scipy.misc import imread, imsave
from scipy.ndimage import label
from os import listdir
from os.path import isfile, join

pi_4 = 4*math.pi

def draw_hist(img):
    ax = plt.subplot(3,1,1)
    ax.set_title('H')
    hist_hue = cv2.calcHist([img], [0], None, [180], [0,180])
    ax.plot(hist_hue)

    ax = plt.subplot(3,1,2)
    ax.set_title('S')
    hist_hue = cv2.calcHist([img], [1], None, [256], [0,255])
    ax.plot(hist_hue)

    ax = plt.subplot(3,1,3)
    ax.set_title('V')
    hist_hue = cv2.calcHist([img], [2], None, [256], [0,255])
    ax.plot(hist_hue)
    plt.ioff()
    plt.show()
    plt.ion()
    #plt.show()

def draw_hue(img):
    ax = plt.subplot(1,1,1)
    ax.set_title('H')
    hist_hue = cv2.calcHist([img], [0], None, [99], [1,100])
    ax.plot(hist_hue)
    plt.ioff()
    plt.draw()

def find_hue_mid(img):
    hist_hue = cv2.calcHist([img], [0], None, [99], [1,100])
    hist_hue = np.array([i[0] for i in hist_hue.tolist()])
    b = (np.diff(np.sign(np.diff(hist_hue))) > 0).nonzero()[0] + 1
    print b
    # Find non-zero min
    idx = min([(i,hist_hue[i]) for i in b if hist_hue[i] > max(hist_hue)/100], key=lambda x: x[1])
    print idx
    return idx[0]

def split_into_rgb_channels(image):
    
    red = image[:,:,2]
    green = image[:,:,1]
    blue = image[:,:,0]
    redgreen = red + green
    
    return red, green, redgreen, blue


def convert_rgb_channels_into_grays(red,green,blue):
    
    grayr = cv2.cvtColor(red,cv2.COLOR_RGB2GRAY)
    grayg = cv2.cvtColor(green,cv2.COLOR_RGB2GRAY)
    grayb = cv2.cvtColor(blue,cv2.COLOR_RGB2GRAY)
    
    return grayr, grayg, grayb
 

def segment_on_dt(img):
    border = img - cv2.erode(img, None)

    dt = cv2.distanceTransform(255 - img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, 100, 255, cv2.THRESH_BINARY)

    lbl, ncc = label(dt)
    lbl[border == 255] = ncc + 1

    lbl = lbl.astype(np.int32)
    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), lbl)
    lbl[lbl < 1] = 0
    lbl[lbl > ncc] = 0

    lbl = lbl.astype(np.uint8)
    lbl = cv2.erode(lbl, None)
    lbl[lbl != 0] = 255
    return lbl   
    
def find_circles(lbl):

    circles, obj_centre = [], []
    contours, _ = cv2.findContours(lbl,
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        c = c.astype(np.int64) # XXX OpenCV bug.
        area = cv2.contourArea(c)
        
        if 3 < area < 500:
            arclen = cv2.arcLength(c, True)
            circularity = (pi_4 * area) / (arclen * arclen)
            if circularity >= 0: # XXX Yes, pretty low threshold.
                circles.append(c)
                box = cv2.boundingRect(c)
                obj_centre.append((box[0] + (box[2] / 2), box[1] + (box[3] / 2)))

    return circles, obj_centre, contours


def track_centre(objcentre, newdata):
    for i in xrange(len(objcentre)):
        ostr, oc = objcentre[i]
        best = min((abs(c[0]-oc[0])**2+abs(c[1]-oc[1])**2, j)
                for j, c in enumerate(newdata))
        j = best[1]
        if i == j:
            objcentre[i] = (ostr, new_centre[j])
        else:
            print "Swapping %s <-> %s" % ((i, objcentre[i]), (j, objcentre[j]))
            objcentre[i], objcentre[j] = objcentre[j], objcentre[i]

imgdir = '../images'    
images = [ '%s' % f for f in listdir(imgdir) ]

myc = 0

obj_centre = None
for image in images:
    # embed()
    myc = myc+1

    if len(sys.argv)>1:
        filename = sys.argv[1]
    else:
        filename = join(imgdir, image)

    img = cv2.imread(filename,0) # color =1, gray = 0, alpha = -1)
    imgcol = cv2.imread(filename)
    imgcol1 = imgcol.copy()
    
    # imgcol[:,:,2] = 0 
    colchannles = []
    colchannels = split_into_rgb_channels(imgcol)
    #gr, gg, gb = convert_rgb_channels_into_grays(red,green,blue)
    # img = cv2.cvtColor(imgcol,cv2.COLOR_BGR2GRAY) 
    imgcol = cv2.cvtColor(imgcol,cv2.COLOR_BGR2RGB)	
    imgcol2 = imgcol.copy()
    if img==None:
        print "cannot open ",filename

    else:
        
        for j in xrange(0,2):

            cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

            img = colchannels[j]
            blur = cv2.GaussianBlur(img,(3,3),0)
            
            # convert to bgr to hsv
            #imgcol1 = cv2.GaussianBlur(imgcol1,(3,3),0)
            hsv = cv2.cvtColor(imgcol1, cv2.COLOR_BGR2HSV)
           
            #draw_hue(hsv)
            # define range of colors in HSV
            lower_red1 = np.array([155,100,50], dtype=np.uint8)
            upper_red1 = np.array([180,255,255], dtype=np.uint8)
            lower_red2 = np.array([0,90,40], dtype=np.uint8)
            upper_red2 = np.array([10,255,255], dtype=np.uint8)
            lower_green = np.array([50,100,50], dtype=np.uint8)
            upper_green = np.array([80,255,255], dtype=np.uint8)
            lower_aqua = np.array([91,110,100], dtype=np.uint8)
            upper_aqua = np.array([115,255,255], dtype=np.uint8)
            lower_gold = np.array([12,10,25], dtype=np.uint8)
            upper_gold = np.array([40,130,255], dtype=np.uint8)
            lower_yellow = np.array([15,131,50], dtype=np.uint8)
            upper_yellow = np.array([28,255,255], dtype=np.uint8)
             
        
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            mask_aqua = cv2.inRange(hsv, lower_aqua, upper_aqua)
            mask_gold = cv2.inRange(hsv, lower_gold, upper_gold)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            
            #border = img - cv2.erode(img, None)
            #plt.subplot(2,2,2),plt.imshow(border)
            #plt.title('channel %d border' % (j))
            
            #dt = cv2.distanceTransform(255 - img, 2, 3)
            #plt.subplot(2,2,4),plt.imshow(dt)
            #plt.title('dt')
            # find normalized_histogram, and its cum_sum
            hist = cv2.calcHist([blur],[0],None,[256],[0,256])
            hist_norm = hist.ravel()/hist.max()
            Q = hist_norm.cumsum()

            bins = np.arange(256)

            fn_min = np.inf
            thresh = -1

            for i in xrange(1,256):
                p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
                q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
                b1,b2 = np.hsplit(bins,[i]) # weights

                # finding means and variances
                m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
                v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

            # calculates the minimization function
            fn = v1*q1 + v2*q2
            if fn < fn_min:
                fn_min = fn
                thresh = i

            # find otsu's threshold value with OpenCV function
            blur = cv2.erode(blur,None,iterations = 1)
            
            ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            print j, thresh, ret
            labelarray, count = label(otsu)
            print count
            rmax = mh.regmax(img)
            
            
            # noise removal
            kernel = np.ones((1,1),np.uint8)
            opening = cv2.morphologyEx(otsu,cv2.MORPH_OPEN,kernel, iterations = 2)
            
            # erode for fg
            fg = cv2.erode(otsu,None,iterations = 2)
            
            # bg alt method
            bgt = cv2.dilate(otsu,None,iterations = 3)
            retbg, bg = cv2.threshold(bgt,1,128,1)
            
            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)
            
            # find sure foreground area
            #dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
            dist_transform = cv2.distanceTransform(
                opening,cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE)
            #print (dist_transform)
            retsfg, sure_fg = cv2.threshold(
                dist_transform,0.7*dist_transform.max(),255,0)
            
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg,sure_fg)
            
            # marker labelling
            labelarray, fgcount = label(sure_fg)
            
            # add one to all labels so that sure background is no 0, but 1
            labelarray1 = labelarray+1
            
            # now, mark the region of unknown with a zero
            labelarray1[unknown==255] = 0
            
            lbl = labelarray1.astype(np.int32)
            signals = cv2.watershed(
                cv2.cvtColor(unknown, cv2.COLOR_GRAY2RGB),labelarray1)
            lbl[lbl < 1] = 0
            lbl[lbl > fgcount] = 0
            
            lbl = lbl.astype(np.uint8)
            lbl - cv2.erode(lbl, None)
            lbl[lbl !=0] = 255
            
            circles, new_centre, contours = find_circles(lbl)
            
            #cv2.drawContours(imgcol,contours,-1,(255,0,255),1)
            #cv2.waitKey(500) == ord('a')
            whole = mh.segmentation.gvoronoi(new_centre)
            
            if obj_centre is None:
                obj_centre = [(str(i + 1), c) for i, c in enumerate(new_centre)]
            else:
                obj_centre = [(str(i + 1), c) for i, c in enumerate(new_centre)]
                #track_centre(obj_centre, new_centre)
            
               
            print '%s, RGB shape (rows,cols,ch): %s, img.size: %s' % (image, imgcol.shape, img.size)
            mycol = [(255,0,0),(0,255,0),(255,255,0)] #rgb
            for i in xrange(len(circles)):
                cv2.drawContours(imgcol, circles, i, mycol[j],1)
                cstr, ccentre = obj_centre[i]
                ccol = imgcol[ccentre] #bgr to rgb
                ccol = np.array(ccol).transpose()
                print '%s, ch=%d, th=%d, ret=%d, count=%d, ccstr=%s, ccentre=%s, b,g,r=%s' % (image, j, thresh, ret, count, cstr, ccentre, ccol)
            cv2.putText(imgcol, cstr, (10,20+20*j), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, mycol[j], 1, cv2.CV_AA)
                
            #cv2.imshow("result", imgcolz)
            #cv2.waitKey(0) == ord('a')
            #plt.figure(figsize=(8,8))
            imgcolz = cv2.cvtColor(imgcol, cv2.COLOR_BGR2RGB)
            
            plt.subplot(3,2,1),plt.imshow(imgcol2)
            plt.title('%s-input ch%d' % (filename, j))
             
            plt.subplot(3,2,2),plt.imshow(mask_red)
            plt.title('ch%d mask_red' % (j))
            plt.subplot(3,2,3),plt.imshow(mask_green)
            plt.title('ch%d mask green' % (j))
            plt.subplot(3,2,4),plt.imshow(mask_aqua)
            plt.title('ch%d mask aqua' % (j))
            plt.subplot(3,2,5),plt.imshow(mask_gold)
            plt.title('ch%d mask gold' % (j))
            plt.subplot(3,2,6),plt.imshow(mask_yellow)
            plt.title('ch%d mask yellow' % (j))
            #plt.subplot(2,2,3),plt.imshow(labelarray1)
            #plt.title('ch%d labelarray1' % (j))
            #plt.subplot(2,2,4),plt.imshow(mh.overlay(img,rmax))
            #plt.title('ch%d mh.rmax' % (j))
            #plt.subplot(2,2,4),plt.imshow(opening)
            #plt.title('ch%d opening' % (j))
            plt.ion()
            plt.draw()
            
            cv2.imshow(image,imgcolz)
            cv2.waitKey(0) == ord('a')
            cv2.destroyAllWindows()

            # Python: cv2.HoughCircles(
                # image, method, dp, minDist[, circles[, param1[, param2 18
                # [, minRadius 10 [, maxRadius]]]]])
            circles = None
            #circles = cv2.HoughCircles(
            #   img, cv2.cv.CV_HOUGH_GRADIENT,
            #   1, 12, np.array([]), param1=20, param2=2, minRadius=0, maxRadius=5
            #)

            if not (circles is None):
                a, b, c = circles.shape 
                # embed()        
                # for i in circles[0,:]: 

                for i in range(b):
                    # draw the outer circle
                    #cv2.circle(imgcol,(circles[0][i][0], circles[0][i][1]),
                     #          circles[0][i][2], (0,0,255), 1)
                    cv2.circle(cimg,(circles[0][i][0], circles[0][i][1]),
                               circles[0][i][2], (0,0,255), 1)
                    # draw the centre of the circle
                    #cv2.circle(imgcol2, (circles[0][i][0], circles[0][i][1]),
                        #2, (0,255,0), 1)   
                    # draw the centre of the circle
                    #cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

                    pad = 5
                    size = 32

                    # size = i[2]
                    xstart = i-size-pad
                    xfinish = i+size+pad
                    ystart = i-size-pad
                    yfinish = i+size+pad

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
                    #embed() 

                    cropImg = imgcol[ystart:yfinish, xstart:xfinish]


                    val = (cropImg[:,:,2] - cropImg[:,:].mean()).mean()
                    val = 11 
                    #embed()

                    print(filename, 'i-%d-%d-%d-%d.jpg' %
                          (circles[0][i][0], circles[0][i][1], circles[0][i][2], val))

                    if abs(val) > 10.0:
                        #imsave('output/%d-%d-%d-%d.jpg' %
                             # (c,xstart,ystart,int(val*100)),cropImg)
                        imsave('output/%d-%d-%d-%d.jpg' % 
                               (circles[0][i][0], circles[0][i][1], 
                                circles[0][i][2], val), 
                               imgcol
                              ) 

            #cv2.imshow(image,imgcolz)
            #cv2.waitKey(0) == ord('a')
            #cv2.destroyAllWindows()

            #plt.subplot(2,2,4)
            #plt.title('Detected circles')
            #plt.imshow(cimg)
            #plt.ion()
            #plt.draw()