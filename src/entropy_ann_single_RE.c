/*
 *  entropy_ann_single_RE.c
 *  
 *  to compute relative entropy with k-nn algorithms
 *  using ANN library : http://www.cs.umd.edu/~mount/ANN/
 *
 *  Created by Nicolas Garnier on 2012/02/25.
 *  Copyright 2012-2021 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2012-02-28 : fork from entropy_nns.c; include and use of ANN library
 *  2012-05-03 : Theiler correction properly implemented
 *  2013-06-20 : forked into entropy_ann_mask.c for masking + slight improvements on tests
 *  2017-11-29 : renamed "search_ANN" as "search_ANN_internal" for future extensions
 *  2019-01-21 : added test for distance==0 inside ANN_wrapper.c (returned value!=0 if problem)
 *               and rewritten all tests for (nb_errors>=npts)
 *  2019-01-22 : rewritten some ANN functions, and cleaned up a little (tests moved back here)
 *  2020-02-26 : new management of "nb_errors" (via global variables)
 *  2021-12-14 : extracted from "entropy_ann.c"
 */


#include <math.h>        /* for fabs and log */
#include <string.h>
#include <gsl/gsl_sf.h>  /* for psi digamma function */ 

#include "library_commons.h"        // definitions of nb_errors, and stds
#include "library_matlab.h"         // compilation for Matlab
#include "ANN_wrapper.h"            // for ANN library (in C++)
#include "math_tools.h"

#define noDEBUG	    // for debug information, replace "noDEBUG" by "DEBUG"
#define noDEBUG_EXPORT
#define LOOK 17 	// for debug also (of which point(s) will we save the data ?)



/**************************************************************************************
 * computes relative Shannon entropy H(x||y), using nearest neighbor statistics
 * this is derived from: Leonenko et al, Annals of Statistics 36 (5) pp2153–2182 (2008)
 * page 2163
 *
 * this version is for n-dimentional systems, and uses ANN library with kd-tree
 *
 * this version does not support embedding per se, but can be used by a wrapper which
 * embedds the data (the function "compute_relative_entropy_ann()" being exactly this)
 *
 * x   contains the data, distributed from f(x). x is of size n*nx
 * nx  is the number of observations of x (number of points in time)
 * y   contains the data, distributed from g(y). y is of size n*ny
 * ny  is the number of observations of y
 * n   is the dimensionality (of both x and y, should be the same)
 * k   is the number of neighbors to consider
 *
 * data is ordered like this :
 * x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)
 *
 * 2017-11-29, fork from "compute_entropy_nd_ann"
 ***************************************************************************************/
double compute_relative_entropy_2xnd_ann(double *x, int nx, double *y, int ny, int n, int k)
{   register int i, j;
    double *x_central;
    double epsilon;
    double h=0.00;

    x_central = (double*)calloc(n, sizeof(double));
    init_ANN(ny, n, k, SINGLE_TH);
    create_kd_tree(y, ny, n);
    nb_errors_local=0; 
    
    for (i=0; i<nx; i++)
    {   for (j=0; j<n; j++) x_central[j] = x[i + j*nx];
          
        epsilon = ANN_find_distance_ex(x_central, n, k, 0); // 2019-01-22 
                                            // 2021-12-01: thread index 0
        if (epsilon<=0) nb_errors_local++;
        else   h = h + log(epsilon);
    }

    if (nb_errors_local>=nx) h=my_NAN;
    else
    {   h = h/(double)(nx-nb_errors_local); /* normalisation de l'esperance */
     
        // normalisation :
        h = h*(double)n;
        h = h + gsl_sf_psi_int(ny) - gsl_sf_psi_int(k);     // 2019-03-18: formula "à la Kraskov"
                                                            // 2025-02-12: removed -nb_errors_local from the first psi function
//        h = h + log((double)ny-1.0) - gsl_sf_psi_int(k);          // 2020-02-19: formula from Leonenko et al (XIII.30)
        h = h + (double)n*log((double)2.0);     /* notre epsilon est le rayon, et pas le diametre de la boule */
    }
    
    /* free pointers : */
    free(x_central);
    free_ANN(SINGLE_TH);
    last_npts_eff_local = nx-nb_errors_local; 
    return(h);
} /* end of function "compute_relative_entropy_2xnd_ann" *************************************/
