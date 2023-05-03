// timings.c
//
// routines for threads time measurements
// 2021-12-09, N.B.G.

#include "timings.h"

__thread struct timespec t1;
__thread struct timespec t2;

__thread struct timespec t1_in; // pour imbriquer les timers
__thread struct timespec t2_in;


void tic(void)
{   clock_gettime(MY_CLOCK, &t1);
}

void toc(double *elapsed)
{   double tmp=0;
    clock_gettime(MY_CLOCK, &t2);
    tmp  = *elapsed;
    tmp += (t2.tv_sec  - t1.tv_sec);
    tmp += (t2.tv_nsec - t1.tv_nsec) / 1000000000.0;
    *elapsed = tmp;
}    

void tic_in(void)
{   clock_gettime(MY_CLOCK, &t1_in);
}

void toc_in(double *elapsed)
{   double tmp=0;
    clock_gettime(MY_CLOCK, &t2_in);
    tmp  = *elapsed;
    tmp += (t2_in.tv_sec  - t1_in.tv_sec);
    tmp += (t2_in.tv_nsec - t1_in.tv_nsec) / 1000000000.0;
    *elapsed = tmp;
}    
