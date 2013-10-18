#!/usr/bin/python

import numpy 
import avxmath
import time
from numpy import random, pi

COUNT=20

x = 100*random.random(10000001)
#x = x.reshape(100,100000)
t = time.time()
for i in range(COUNT):
    y1 = numpy.sin(x)
duration1 = time.time() - t
print "numpy.sin %f sec" % duration1

t = time.time()
for i in range(COUNT):
    y2 = avxmath.sin(x)
duration2 = time.time() - t
print "avxmath.sin %f sec" % duration2

print "max difference is %lg" % (y2 - y1).max()
print "speedup is %lg" % (duration1/duration2)
