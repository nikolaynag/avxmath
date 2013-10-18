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
typedef struct {
    double (*mathfunc)(double);
    vdouble (*avxfunc)(vdouble);
    char name[10];
    char docstring[256];
} tandem;

tandem sin_tandem = {&sin, &xsin, "sin", "AVX-accelerated float64 sinus calculation"};
tandem cos_tandem = {&cos, &xcos, "cos", "AVX-accelerated float64 cosinus calculation"};
tandem exp_tandem = {&exp, &xexp, "exp", "AVX-accelerated float64 exponent calculation"};

static tandem *tandems[] = {
    &sin_tandem, &cos_tandem, &exp_tandem,
};

static void double_xloop(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    tandem *f = (tandem *)data;
    double (*mathfunc)(double) = f->mathfunc;
    vdouble (*avxfunc)(vdouble) = f->avxfunc;
    for(i = 0; i < n % VECTLENDP; i++)
    {
        *((double *)out) = (*mathfunc)(*(double *)in);
        in += in_step;
        out += out_step;
    }
    if(i == n) return;
    if(in_step != sizeof(double) || out_step != sizeof(double))
    {
        int i;
        out = args[1];
        for(i = 0; i < n; i++)
        {
            *((double *)out) = NAN;
            out += out_step;
        }
        return;
    } 
    else 
    {
    	vdouble a;
        double *in_array = (double *)in;
        double *out_array = (double *)out;
        for(i = 0; i < n; i += VECTLENDP)
        {
            a = vloadu(in_array + i);
    	    a = (*avxfunc)(a);
	    	vstoreu(out_array + i, a);    
        }
   }
}
/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&double_xloop};
/* These are the input and return dtypes of our functions.*/
char types[2] = {NPY_DOUBLE, NPY_DOUBLE};

static void register_avx_tandems(PyObject *m)
{
    int i;
    PyObject *f, *d;
    for(i = 0; i < sizeof(tandems)/sizeof(tandems[0]); i++)
    {
        
        f = PyUFunc_FromFuncAndData(funcs, tandems + i, types, 1, 1, 1,
                                    PyUFunc_None, tandems[i]->name,
                                    tandems[i]->docstring, 0);
        d = PyModule_GetDict(m);
        PyDict_SetItemString(d, tandems[i]->name, f);
        Py_DECREF(f);
    }
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
    register_avx_tandems(m);
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
    register_avx_tandems(m);
}
#endif
