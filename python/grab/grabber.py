import urllib2
import time
from itertools import product
from IPython import embed

start_x = 14160
start_y = 768
finish_x = 25440
finish_y = 29440
img_x = 1024
img_y = 1024
quality = 80
 
for x, y in product(xrange(start_x, finish_x, img_x), xrange(start_y, finish_y, img_y)):
    url = "http://slides.virtualpathology.leeds.ac.uk/Research_4/Slide_Library/hmds/Set_5/162501.svs?%d+%d+%d+%d+1+%d+S" % (x, y, img_x, img_y, quality)
    data = urllib2.urlopen(url)
    with open('data/1-%d-%d' % (x, y), 'w') as f:
      while True:
        chunk = data.read(1024*1024*8)
        if not chunk: break
        f.write(chunk)
    print "Grabbed: %d, %d" % (x,y)
    time.sleep(1) 
