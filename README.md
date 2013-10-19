avxmath
=======

SIMD-optimized mathematical functions for numpy. Numpy module is based on SLEEF
library (http://shibatch.sourceforge.net/). At this moment it supports only AVX
and provides four functions:
sin - numpy ufunc for sine calculation
cos - numpy ufunc for cosine calculation
exp - numpy ufunc for exponent calculation
imexp - calculate real and imaginary part of exponent from pure imaginary argument (uses sincos - simultanious sine and cosine calculation).
