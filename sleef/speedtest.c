#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "sleefsimd.h"

#define TESTSIZE (VECTLENDP*10000000)
double s[TESTSIZE];
double r1[TESTSIZE];
double r2[TESTSIZE];
#define COUNT 10
int main(int argc, char *argv[])
{
    int k, i;
    clock_t t1, t2;
    double time1, time2;
    double max, rmax;

    srandom(time(NULL));
    for(i = 0; i < TESTSIZE; i++) 
    {
        s[i] = random()/(double)RAND_MAX*20000-10000;
    }

    printf("Testing sin, %d values\n", TESTSIZE*COUNT);
    t1 = clock();
    for(k = 0; k < COUNT; k++)
    {
        for(i = 0; i < TESTSIZE; i++) 
        {
            r1[i] = sin(s[i]);
        }
    }
    t2 = clock();
    time1 = (double)(t2 - t1)/CLOCKS_PER_SEC;
    printf("Finish sin, spent time = %lg sec\n\n", time1);

    printf("Testing xsin\n");
    t1 = clock();
    for(k = 0; k < COUNT; k++)
    {
        for(i = 0; i < TESTSIZE; i += VECTLENDP) 
        {
            vdouble a = vloadu(s+i);
            a = xsin(a);
            vstoreu(r2+i, a);
        }
    }
    t2 = clock();
    time2 = (double)(t2 - t1)/CLOCKS_PER_SEC;
    printf("Finish xsin, spent time = %lg sec\n\n", time2);

    printf("Speed ratio: %lf\n", time1/time2);

    max = r1[0] - r2[0];
    rmax = (r1[0] - r2[0])/r1[0];
    for(i = 0; i < TESTSIZE; i++) 
    {
        double delta = (r1[i] - r2[i]);
        if(abs(delta) > abs(max)) max = delta;
        delta = (r1[i] - r2[i])/r1[i];
        if(abs(delta) > abs(max)) rmax = delta;
    }

    printf("Max absolute delta: %lg, relative delta %lg\n", max, rmax);
    return 0;
}
