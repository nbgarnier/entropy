/*
 *  entropy_ann_single_entropy.c
 *  
 *  to compute Shannon entropy with k-nn algorithms
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
 *  2021-12-17 : extracted from "entropy_ann.c"
 */

#include <math.h>               // for fabs and log
#include <string.h>
#include <gsl/gsl_sf.h>         // for psi digamma function

#include "library_commons.h"    // definitions of nb_errors, and stds
#include "library_matlab.h"     // compilation for Matlab
#include "ANN_wrapper.h"        // for ANN library (in C++)
#include "math_tools.h"

#define noDEBUG	    // for debug information, replace "noDEBUG" by "DEBUG"
#define noDEBUG_EXPORT
#define LOOK 17 	// for debug also (of which point(s) will we save the data ?)



/****************************************************************************************/
/* computes Shannon entropy, using nearest neighbor statistics (2004)                   */
/* this is derived from PRE 69 066138 (2004)                                            */
/*                                                                                      */
/* this version is for n-dimentional systems, and uses ANN library with kd-tree         */
/*                                                                                      */
/* this version does not support embedding per se, but can be used by a wrapper which   */
/* embedds the data (the function "compute_entropy_ann()" being exactly this)           */
/*                                                                                      */
/* x   contains all the data, which is of size n*nx                                     */
/* nx  is the number of points in time                                                  */
/* n   is the dimensionality                                                            */
/* k   is the number of neighbors to consider                                           */
/*                                                                                      */
/* data is ordered like this :                                                          */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)               */
/* 2012-02-27, fork from "compute_entropy_ann"                                          */
/****************************************************************************************/
double compute_entropy_nd_ann(double *x, int npts, int n, int k)
{	register int i;
    double epsilon=0.0;
	double h=0.00;
    
//    printf(" ANN state before init : %d\n", get_ANN_ALLOW_SELF_MATCH());
	init_ANN(npts, n, k, SINGLE_TH); 	
//    printf(" ANN state before tree : %d\n", get_ANN_state());
    create_kd_tree(x, npts, n);
//    printf(" ANN state after tree  : %d\n", get_ANN_ALLOW_SELF_MATCH());
    nb_errors_local=0;
    
//    printf("[compute_entropy_nd_ann] : init OK\n");
    for (i=0; i<npts; i++)
	{ //  search_ANN_internal(i, n, k, &l, epsilon_z);
        epsilon = ANN_find_distance_in(i, n, k, 0); // 2019-01-22
        // 2021-12-01: new parameterin the function call above, to specify the thread index

        if (is_zero(epsilon)) // 2020-07-17 replaced ==0 by (epsilon<2*DBL_MIN) and then by this
        {  nb_errors_local++;
//            printf("epsilon = %f\n", epsilon*1e14);
#ifdef DEBUG
           printf("[compute_entropy_nd_ann] point %d (%f) : couille !!\n", i, x[i]);
#endif           
        }
        else /* estimateur de l'entropie : esperance de la grandeur suivante : */
        {  h += log(epsilon);
//           fprintf(fe,"%f\n",epsilon);
        }
	}
    
    if (nb_errors_local>=npts) h=my_NAN;   // big trouble
    else // we can get an estimate
    {   h /= (double)(npts-nb_errors_local); /* normalisation de l'esperance */
	
        /* normalisation : */
        h *= (double)n;
        h += gsl_sf_psi_int(npts-nb_errors_local) - gsl_sf_psi_int(k);
        h += (double)n*log((double)2.0);	/* notre epsilon est le rayon, et pas le diametre de la boule */
    }
//    printf("nb errors local : %d, DBL_MIN = %g\n", nb_errors_local, DBL_MIN*1e13);
    
	/* free pointers de taille n=dimension de l'espace : */
	free_ANN(SINGLE_TH);
	last_npts_eff_local = npts-nb_errors_local;
//	printf("[compute_entropy_nd_ann] : %d effective points\n", last_npts_eff_local);
	return(h);
} /* end of function "compute_entropy_nd_ann" *************************************/


