#!/usr/bin/python

import numpy 
import avxmath
import time
from numpy import random, pi

COUNT=10

functions = [
    ("sin", numpy.sin, avxmath.sin, (-1e4, 1e4)),
    ("cos", numpy.cos, avxmath.cos, (-1e4, 1e4)),
    ("exp", numpy.exp, avxmath.exp, (-7e2, 7e2)),
]

def test_func(arg, name, npfunc, avxfunc):
    t = time.clock()
    for i in xrange(COUNT):
        y1 = npfunc(x)
    duration1 = time.clock() - t
    print "numpy.%s %f sec" % (name, duration1)

    t = time.clock()
    for i in xrange(COUNT):
        y2 = avxfunc(x)
    duration2 = time.clock() - t
    print "avxmath.%s %f sec" % (name, duration2)

    delta = y2 - y1;
    rdelta = delta/y1;
    print "Max absolute difference is %lg, relative %lg" % (
            delta[abs(delta).argmax()], rdelta[abs(rdelta).argmax()])
    print "Speedup is %lg" % (duration1/duration2)

r = random.random(40000000);
for name, npfunc, avxfunc, trange in functions:
    x1, x2 = trange
    x = x1 + r*(x2 - x1)
    test_func(x, name, npfunc, avxfunc)
    print

x = -1e4 + r*2e4;
ys1, yc1 = numpy.sin(x), numpy.cos(x)
ys2, yc2 = avxmath.sincos(x)
delta = ys2 - ys1
rdelta = delta/ys1
print "sincos sin: max absolute difference is %lg, relative %lg" % (
            delta[abs(delta).argmax()], rdelta[abs(rdelta).argmax()])
delta = yc2 - yc1
rdelta = delta/yc1
print "sincos cos: max absolute difference is %lg, relative %lg" % (
            delta[abs(delta).argmax()], rdelta[abs(rdelta).argmax()])

