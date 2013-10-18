import numpy 
import avxmath
import time
from numpy import random, pi

COUNT=50

x = 10*pi*random.random(10000000)
t = time.time()
for i in range(COUNT):
    y = numpy.sin(x)
print "numpy.sin %f sec" % (time.time() - t)

t = time.time()
for i in range(COUNT):
    y = avxmath.sin(x)
print "avxmath.sin %f sec" % (time.time() - t)


