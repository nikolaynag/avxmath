#!/usr/bin/python

import numpy 
import avxmath
import time
from numpy import random, pi

COUNT=1

functions = [
    ("sin", numpy.sin, avxmath.sin),
    ("cos", numpy.cos, avxmath.cos),
    ("exp", numpy.exp, avxmath.exp),
]

def test_func(name, npfunc, avxfunc):
    x = random.random(10000001)
    #x = x.reshape(100,100000)
    t = time.time()
    for i in range(COUNT):
        y1 = npfunc(x)
    duration1 = time.time() - t
    print "numpy.%s %f sec" % (name, duration1)

    t = time.time()
    for i in range(COUNT):
        y2 = avxfunc(x)
    duration2 = time.time() - t
    print "avxmath.%s %f sec" % (name, duration2)

    print "max difference is %lg" % (y2 - y1).max()
    print "speedup is %lg" % (duration1/duration2)

for name, npfunc, avxfunc in functions:
    test_func(name, npfunc, avxfunc)
    print 

x = random.random(10000001)
ys1, yc1 = numpy.sin(x), numpy.cos(x)
ys2, yc2 = avxmath.sincos(x)

print "sincos: max difference sin is %lg" % abs(ys2 - ys1).max()
print "sincos: max difference cos is %lg" % abs(yc2 - yc1).max()


