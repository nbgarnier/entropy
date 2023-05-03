/*
 *  entropy_ann_threads.c
 *
 *  Created by Nicolas Garnier on 2021/11/25.
 *  Copyright 2021 ENS-Lyon - CNRS. All rights reserved.
 *
 *
 *  2021-12-20: new function "adapt_cores_number"
 */
#include <unistd.h>  // to probe the nb of processors

#include "library_matlab.h"         // compilation for Matlab
#include "entropy_ann_threads.h"    // for globals
#include "library_matlab.h"

// single-thread or multi-thread process:
int USE_PTHREAD = 2;    // a variable that can be tuned to use or not multi-threading
                        // 0 : single thread (no pthread at all) (old default, before 2021-12-21)
                        // 1 : multithread, with fixed nb of threads
                        // 2 : multithread, with adaptative nb of threads (default, since 2021-12-21)
                        // 4 : double-layer multithreading (new 2022-01-11) (for images only) 
int _n_cores = -1;      // a variable that can be tuned to select the nb of threads
                        // default = -1 = auto-detect
    

int get_multithreading_state(int verbosity)
{   if (verbosity>0) 
    {    printf("[info] internal variable USE_PTHREAD=%d ", USE_PTHREAD);
         if      (USE_PTHREAD==0) printf("(no multithreading)\n");
         else if (USE_PTHREAD==1) printf("(multithreading with %d threads)\n", _n_cores);
         else if (USE_PTHREAD==2) printf("(multithreading with self-adapted %d threads)\n", _n_cores);
         else if (USE_PTHREAD==4) printf("(double multithreading for images)\n");
    }
    return(USE_PTHREAD); 
}

void set_multithreading_state(int do_mp)
{   if (do_mp>=0) 
         USE_PTHREAD = do_mp;   // we fully transmit the indicated prescription
    else USE_PTHREAD = 2;       // default to automatic multithreading
    return; 
}



// get and/or print the nb of cores:
// depending on parameter "get_what":
// == GET_CORES_AVAILABLES : returns silently NCORES
// == GET_CORES_SELECTED   : returns silently _n_cores
// == something else       : returns _n_cores, and print some infos
int get_cores_number(int get_what)
{   if      (get_what==GET_CORES_AVAILABLE) return((int)NCORES);
    else if (get_what==GET_CORES_SELECTED)  return(_n_cores);
    else
    {   printf("This system has %d processors configured and currently %d processors available.\n", 
                (int)sysconf(_SC_NPROCESSORS_CONF), (int)sysconf(_SC_NPROCESSORS_ONLN));
        printf("The following information is not working on macos, but may on Linux: %d configured / %d available.\n", 
                (int)_SC_NPROCESSORS_CONF, (int)_SC_NPROCESSORS_ONLN);
        printf("I will use %d threads (-1 means default value %d on this computer)\n", _n_cores, NCORES);
    }
    return(_n_cores);
}

void set_cores_number(int n)
{   if (n>0)    _n_cores=n;
    else        _n_cores=NCORES; // auto-detect
    return;
}


/************************************************************************/
// some extensive tests conducted on 2021-12-20 revealed that it may not be
// time-efficient to have a large number of threads when the number of points 
// is not very large.
// As a consequence, I strongly suggest to adapt the nb of threads depending on 
// the number of points to analyze.
// The following function does exactly this, following an empirical rule from
// measurement on my computer (Macbook Pro with officialy 12 cores available)
// The number 117 results from a balance between the pthread gain and the 
// overhead when copying pointers contents.
//
// npts_eff : nb of points to analyze
// returned : "best" nb of cores 
// 
// 2021-12-20 : if (USE_PTHREAD==2) then this function changes the nb of threads.
// 2022-12-13 : now this function only returns the "optimal" value, but does not change it 
//              it is up to the user to change it afterwards
// 
/************************************************************************/
int best_cores_number(int npts_eff)
{   register int best_nb=1;

    best_nb  = npts_eff/ANN_PTS_PER_THREAD;
    if (best_nb<=0) best_nb=1;      // stupid, but required
    if (USE_PTHREAD==2)
    {   best_nb = (best_nb > NCORES) ? NCORES : best_nb; 
    } 
    return(best_nb); // this allows some log/tracing if necessary
}
