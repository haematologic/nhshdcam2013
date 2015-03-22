from IPython import embed
import cv2
import numpy as np
import time
import sys
import matplotlib
matplotlib.use('TKAgg')
# import mahotas as mh
from matplotlib import pyplot as plt
from scipy.misc import imread, imsave
from os import listdir
from os.path import isfile, join

def split_into_rgb_channels(image):
    
    red = image[:,:,2]
    green = image[:,:,1]
    blue = image[:,:,0]
    
    return red, green, blue


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
    
    
images = [ '../images/%s' % f for f in listdir('../images') ]

c = 0

for image in images:
    # embed()
    c = c+1

    if len(sys.argv)>1:
        filename = sys.argv[1]
    else:
        filename = image

    img = cv2.imread(filename,0) # color =1, gray = 0, alpha = -1)
    imgcol = cv2.imread(filename)
    # imgcol[:,:,2] = 0 
    red, green, blue = split_into_rgb_channels(imgcol)
    #gr, gg, gb = convert_rgb_channels_into_grays(red,green,blue)
    # img = cv2.cvtColor(imgcol,cv2.COLOR_BGR2GRAY) 
    imgcol = cv2.cvtColor(imgcol,cv2.COLOR_BGR2RGB)	
     
    if img==None:
        print "cannot open ",filename

    else:
        #img = cv2.medianBlur(img,5)
        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #plt.subplot(2,2,1)
        #plt.title('%s' % (filename))
        #plt.imshow(imgcol)
        #plt.subplot(2,2,2)
        #plt.title('green')
        #plt.imshow(green)
        img = green
        #img = cv2.medianBLue(img,5)
        #img = cv2.GaussianBlur(img, (5, 5,), 2)
        #img1 = img
        #edges = img_gray - cv2.erode(img_gray, None)
        #_, bin_edge = cv2.threshold(edges, 0, 255, cv2.THRESH_OTSU)
        #height, width = bin_edge.shape
        
        #ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        #th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        #            cv2.THRESH_BINARY,11,2)
        #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #            cv2.THRESH_BINARY,11,2)

        #plt.subplot(2,2,1),plt.imshow(img,'gray')
        #plt.title('input image')
        #plt.subplot(2,2,2),plt.imshow(th1,'gray')
        #plt.title('Global Thresholding')
        #plt.subplot(2,2,3),plt.imshow(th2,'gray')
        #plt.title('Adaptive Mean Thresholding')
        #plt.subplot(2,2,4),plt.imshow(th3,'gray')
        #plt.title('Adaptive Gaussian Thresholding')
        #plt.subplot(2,2,2),plt.imshow(edges)
        #plt.title('edges')
        #plt.subplot(2,2,3),plt.imshow(bin_edge)
        #plt.title('bins')
        
        #mask = np.zeros((height+2, width+2), dtype=np.uint8)
        #cv2.floodFill(bin_edge, mask, (0, 0), 255)
        
        #plt.subplot(2,2,4),plt.imshow(bin_edge)
        #plt.title('bins-green')
        # global thresholding
        
        
        #ret1,th1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)

        # Otsu's thresholding
        #ret2,th2 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Otsu's thresholding after Gaussian filtering
        #blur = cv2.GaussianBlur(img1,(5,5),0)
        #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # plot all the images and their histograms
        #titles = ['img1','histogram1','th1',
        #          'img1','histogram2','th2',
        #          'blur','histogram3','th3']

        #for i in xrange(3):
        #    plt.subplot(3,3,i*3+1),plt.imshow(eval(titles[i*3]),'gray')
        #    plt.title(titles[i*3])
        #    plt.subplot(3,3,i*3+2),plt.hist(eval(titles[i*3]).ravel(),256)
        #    plt.title(titles[i*3+1])
        #    plt.subplot(3,3,i*3+3),plt.imshow(eval(titles[i*3+2]),'gray')
        #    plt.title(titles[i*3+2])

        blur = cv2.GaussianBlur(img,(5,5),0)
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
        ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print thresh,ret
        plt.subplot(1,2,1),plt.imshow(imgcol)
        plt.title('%s-input' % (filename))
        plt.subplot(1,2,2),plt.imshow(otsu)
        plt.title('otsu')
        plt.ion()
        plt.draw()

        # x = segment_on_dt(green)
        # cv2.waitKey(0) == ord('a') 
        # Python: cv2.HoughCircles(
            # image, method, dp, minDist[, circles[, param1[, param2 18
            # [, minRadius 10 [, maxRadius]]]]])

        circles = cv2.HoughCircles(
            img, cv2.cv.CV_HOUGH_GRADIENT,
            1, 12, np.array([]), param1=20, param2=2, minRadius=0, maxRadius=5
        )

        #circles = np.uint16(np.around(circles))
        if not (circles is None):
            a, b, c = circles.shape 
            # embed()        
            # for i in circles[0,:]: 

            for i in range(b):
                # draw the outer circle
                cv2.circle(imgcol,(circles[0][i][0], circles[0][i][1]),
                           circles[0][i][2], (0,0,255), 1)
                cv2.circle(cimg,(circles[0][i][0], circles[0][i][1]),
                           circles[0][i][2], (0,0,255), 1)
                # draw the centre of the circle
                #cv2.circle(imgcol2, (circles[0][i][0], circles[0][i][1]),
                    #2, (0,255,0), 1)   
                # draw the center of the circle
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

                cv2.imshow('detected circles',cimg)
                cv2.waitKey(0) == ord('a')
                cv2.destroyAllWindows()

            #plt.subplot(2,2,4)
            #plt.title('Detected circles')
            #plt.imshow(cimg)
            #plt.ion()
            #plt.draw()