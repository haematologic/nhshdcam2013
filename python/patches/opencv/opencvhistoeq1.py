from os import listdir
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
import sys 
from IPython import embed 
from scipy.misc import imread, imsave 
images = [ '../images/%s' % f for f in listdir('../images') ]
c = 0

for image in images:
    
    c = c+1
    #embed()

    if len(sys.argv)>1:
        filename = sys.argv[1]
    else:
        filename = image
        
    img = cv2.imread(filename,1) # 0 for grayscale, 1 color, -1 alpha 
    img = np.asarray(img[:,:])
    
    cv2.imshow("original",img)
    cv2.waitKey(25)
    embed() 
    hist = cv2.calcHist([img],[2],None,[256],[0,256])

    #convert img to YCR_CB
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)

    #split image to Y, CR, CB
    img2[:, :, 0] = cv2.equalizeHist(img2[:, :, 0])
    #cv2.split(img2,channels)

    #histogram equalization to Y-MATRIX
    #cv2.equalizeHist(channels[0],channels[0])

    #merge this matrix to reconstruct our colored image
    #cv2.merge(channels,img2)

    #convert this output image to rgb	
    rgb = cv2.cvtColor(img2,cv2.COLOR_YCR_CB2BGR)		
    hist2 = cv2.calcHist([rgb],[2],None,[256],[0,256])
    plt.plot(hist)
    plt.plot(hist2)
    plt.show()

    cv2.imwrite('output/%s-%d.jpg' % ('img2',c),rgb)

