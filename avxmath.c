#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include "sleef/sleefsimd.h"
/*
 * avxmath.c
 * This is the C code for avx-accelerated
 * Numpy ufunc for mathematical functions.
 *
 * In this code we only define the ufunc for
 * a single dtype (float64). 
*/

static PyMethodDef AvxmathMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void double_avxsin(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp k, i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double tmp[VECTLENDP];
	vdouble a;
    int chunks = n/VECTLENDP;
    for(k = 0; k < chunks; k++)
    {
        for(i = 0; i < VECTLENDP; i++) {
            tmp[i] = *(double *)in;
            in += in_step;
        }
        a = vloadu(tmp);
	    a = xsin(a);
		vstoreu(tmp, a);    
        for(i = 0; i < VECTLENDP; i++) {
            *((double *)out) = tmp[i];
            out += out_step;
        }
    }

    for (i = VECTLENDP*chunks; i < n; i++) {
        
        *((double *)out) = sin(*(double *)in);
        in += in_step;
        out += out_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&double_avxsin};

/* These are the input and return dtypes of our functions.*/
static char types[2] = {NPY_DOUBLE, NPY_DOUBLE};

static void *data[1] = {NULL};

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
    PyObject *m, *avxsin, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    avxsin = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                    PyUFunc_None, "avxsin",
                                    "avxsin_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "avxsin", avxsin);
    Py_DECREF(avxsin);

    return m;
}
#else
PyMODINIT_FUNC initavxmath(void)
{
    PyObject *m, *avxsin, *d;


    m = Py_InitModule("avxmath", AvxmathMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    avxsin = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                    PyUFunc_None, "avxsin",
                                    "avxsin_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "avxsin", avxsin);
    Py_DECREF(avxsin);
}
#endif
