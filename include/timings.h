// timings.h
//
// routines for threads time measurements
// this offers 1 clock per thread, which can be used sequentially
//
// 2021-12-09, N.B.G.

//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wgnu-statement-expression"
//#pragma GCC diagnostic pop
   
#include <time.h>
#ifdef CLOCK_MONOTONIC
// https://coderedirect.com/questions/56278/c-using-clock-to-measure-time-in-multi-threaded-programs
    #define MY_CLOCK CLOCK_MONOTONIC 
#else
    #define MY CLOCK CLOCK_REALTIME
#endif

extern __thread struct timespec t1;
extern __thread struct timespec t2;

extern __thread struct timespec t1_in;
extern __thread struct timespec t2_in;

void   tic(void);
void   toc(double *elapsed);

void   tic_in(void);
void   toc_in(double *elapsed);

