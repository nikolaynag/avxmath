#include "Python.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include "sleef/sleefsimd.h"
/*
 * avxmath.c
 * This is the C code for avx-accelerated
 * Numpy ufuncs for mathematical functions.
 *
 * In this code we only define the ufuncs for
 * a single dtype (float64). 
 * Copyright 2013 Nikolay Nagorskiy
 *
 * For SIMD-accelerated mathematical functions this module uses SLEEF 
 * library (http://shibatch.sourceforge.net/)
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
*/

static PyMethodDef AvxmathMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */
typedef struct {
    vdouble (*f)(vdouble);
    char name[10];
    char docstring[256];
} avx_func;

avx_func avx_sin = {&xsin, "sin", "AVX-accelerated float64 sinus calculation"};
avx_func avx_cos = {&xcos, "cos", "AVX-accelerated float64 cosinus calculation"};
avx_func avx_exp = {&xexp, "exp", "AVX-accelerated float64 exponent calculation"};

static avx_func *functions[] = {
    &avx_sin, &avx_cos, &avx_exp,
};

static void double_xloop(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    avx_func *func = (avx_func *)data;
    vdouble (*f)(vdouble) = func->f;
    double tmp[VECTLENDP];
    vdouble a;
    int slow_n = n % VECTLENDP;
    if(in_step != sizeof(double) || out_step != sizeof(double))
        slow_n = n;
    for(i = 0; i < slow_n; i += VECTLENDP)
    {
        int j;
        for(j = 0; j < VECTLENDP && i + j < slow_n; j++)
        {
            tmp[j] = *(double *)in;
            in += in_step;            
        }
        a = vloadu(tmp);
        a = (*f)(a);
        vstoreu(tmp, a);
        for(j = 0; j < VECTLENDP && i + j < slow_n; j++)
        {
            *(double *)out = tmp[j];
            out += out_step;
        }        
    }
    if(n > slow_n)
    {
        double *in_array = (double *)in;
        double *out_array = (double *)out;
        for(i = 0; i < n - slow_n; i += VECTLENDP)
        {
            a = vloadu(in_array + i);
        	a = (*f)(a);
	        vstoreu(out_array + i, a);    
        }
    }
}

static void double_sincos(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out1 = args[1], *out2 = args[2];
    npy_intp in_step = steps[0], out1_step = steps[1], out2_step = steps[2];
    double tmp1[VECTLENDP], tmp2[VECTLENDP];
    vdouble a;
    vdouble2 b;
    int slow_n = n % VECTLENDP;
    if(in_step != sizeof(double) || out1_step != sizeof(double) || 
       out2_step != sizeof(double))
    {
        slow_n = n;
    }
    for(i = 0; i < slow_n; i += VECTLENDP)
    {
        int j;
        for(j = 0; j < VECTLENDP && i + j < slow_n; j++)
        {
            tmp1[j] = *(double *)in;
            in += in_step;            
        }
        a = vloadu(tmp1);
        b = xsincos(a);
        vstoreu(tmp1, b.x);    
        vstoreu(tmp2, b.y);    
        for(j = 0; j < VECTLENDP && i + j < slow_n; j++)
        {
            *(double *)out1 = tmp1[j];
            *(double *)out2 = tmp2[j];
            out1 += out1_step;
            out2 += out2_step;
        }        
    }
    if(n > slow_n)
    {
        double *in_array = (double *)in;
        double *out_array1 = (double *)out1;
        double *out_array2 = (double *)out2;
        for(i = 0; i < n - slow_n; i += VECTLENDP)
        {  
            a = vloadu(in_array + i);
    	    b = xsincos(a);
	    	vstoreu(out_array1 + i, b.x);    
	    	vstoreu(out_array2 + i, b.y); 
        }
    }
}

static PyUFuncGenericFunction funcs1[] = {&double_xloop};
static PyUFuncGenericFunction funcs2[] = {&double_sincos};

/* These are the input and return dtypes of our functions.*/
static char types1[] = {NPY_DOUBLE, NPY_DOUBLE};
static char types2[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,};
static void *data2[] = {NULL};

static void register_avx_functions(PyObject *m)
{
    int i;
    PyObject *f, *d;
    d = PyModule_GetDict(m);
    for(i = 0; i < sizeof(functions)/sizeof(functions[0]); i++)
    {
        
        f = PyUFunc_FromFuncAndData(funcs1, (void *)(functions + i), types1, 1, 1, 1,
                                    PyUFunc_None, functions[i]->name,
                                    functions[i]->docstring, 0);
        PyDict_SetItemString(d, functions[i]->name, f);
        Py_DECREF(f);
    }
    f = PyUFunc_FromFuncAndData(funcs2, data2, types2, 1, 1, 2,
                                    PyUFunc_None, "sincos",
                                    "AVX-accelerated simultanious sin and cos computation", 0);
    PyDict_SetItemString(d, "sincos", f);
    Py_DECREF(f);
}

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "avxmath",
    NULL,
    -1,
    AvxmathMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_avxmath(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    import_umath();
    register_avx_functions(m);
    return m;
}
#else
PyMODINIT_FUNC initavxmath(void)
{
    PyObject *m;
    m = Py_InitModule("avxmath", AvxmathMethods);
    if (m == NULL) {
        return;
    }
    import_array();
    import_umath();
    register_avx_functions(m);
}
#endif
