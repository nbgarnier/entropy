/*
 *  entropy_ann_2d.c
 *  
 *  to compute entropies on 2d data (images) with k-nn algorithms
 *  using ANN library : http://www.cs.umd.edu/~mount/ANN/
 *
 *  Forked by Nicolas Garnier on 2020/07/19.
 *  Copyright 2012-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 *
 *  2020-07-19: new wrapper for images
 *  2020-09-01: preparations for struct type parameters
 *  2020-01-05: multithread v1
 */

#include <stdlib.h>
#include <stdio.h>      // for printf
#include <string.h>     // for malloc
#include <math.h>       // for log 
#include <gsl/gsl_statistics_double.h>  // for std of intermediate (pre-processed) data

#include "library_commons.h"   // for is_zero()
#include "library_matlab.h"
#include "ANN_wrapper.h"
#include "samplings.h"
#include "entropy_ann_threads.h"
#include "entropy_ann_single_entropy.h"
#include "entropy_ann_threads_entropy.h" // for the special threaded-function to compute Shannon entropy

#define noDEBUG

// thread arguments and outputs types:
/*struct thread_args
    {   int core;    // keep track of the current thread number/id
        int i_start; // begining of subset of points to work on
        int i_end;   // end      of subset of points
        int n;       // dimensionality of data
        int k;       // nb of neighbors to search for
    };

struct thread_output
    {   int n_eff;
        int n_errors;
        double h_sum;
    };
*/

#ifdef DEBUG
/* 2018-04-11/3: test of memory alignment:                                              */
/* 2020-07-20: this version is for Python images,                                       */
/*             i.e., x refers to 1st dimension, which is the vertical for a matrix,     */
/*                 and y refers to 2nd dimension, which is horizontal                   */
void debug_trace_2d(char *text, double *x, int nx, int ny)
{    register int i, j;
#define NPTS_X 9
#define NPTS_Y 9
     printf("%s - nx=%d, ny=%d\n", text, nx, ny);
     for (j=0; j<(NPTS_Y>ny?ny:NPTS_Y); j++)
     {    printf("  vector %d: [ ", j);
          for (i=0; i<(NPTS_X>nx?nx:NPTS_X); i++) printf("%f ",x[i*ny+j]);
          printf("] along x\n");
     }
     return;
}
#endif


/****************************************************************************************/
/* computes Shannon entropy, using nearest neighbor statistics (Grassberger 2004)	    */
/*                                                                                      */
/* this version is for images, i.e., 2-dimensional scalar or vectorial data    	        */
/*																			            */
/* x        : contains all the data, which is of size (nx,ny)   					    */
/* nx, ny   : the number of points in the X and Y dimensions						    */
/* d        : the dimensionality of the data                                            */
/* p 	    : the embedding dimension                                                   */
/* stride_x,                                                                            */
/* stride_y : the distance between 2 consecutive points for embedding in x and y        */
/* Theiler_x,                                                                           */
/* Theiler_y: the Theiler distance in x and y                                           */
/* N_eff          : the nb of effective points to use                                   */
/* N_realizations : the requested nb oof realizations                                   */
/* k        : the nb of neighbors to be considered										*/
/*																			            */
/* beware the ordering! different from the 1-d multidimensional data!                   */
/* 2-dimensional data is ordered as follows (to follow Python convention)			    */
/* x1(x=0,y=0) ... x1(x=nx-1,y=0),                                                      */
/* x1(x=0,y=1) ... ... x1(x=nx-1,y=ny-1),                                               */
/* x2(x=0,y=0) ... ... x2(x=nx-1,y=ny-1),                                               */
/* ... xd(0,0) ... ..  xd(x=nx-1,y=ny-1)                                				*/
/*																			            */
/* this function is a wrapper to the functions :									    */
/*	- compute_entropy_nd_ann                                                            */
/*																			            */
/* 2020-07-19 : new wrapper function, assuming d=1 for now (scalar images)              */
/* 2020-08-04 : now allowing multi-dimensional images and different strides in x and y  */
/* 2022-01-06 : breaking change in embedding definition                                 */
/* 2022-01-08 : tested OK                                                               */
/* 2022-01-10 : adapted Theiler prescriptions (see brouillon)                           */
/* 2022-01-12 : now also returning entropy of increments (via "method")                 */
/* 2022-05-04 : parameters change, to accomodate Theiler_x,y and N_eff_x,y              */
/* 2022-05-24 : working version, bug corrected                                          */
/****************************************************************************************/
//int compute_entropy_ann_2d(double *x, dimension_parameters dim, embedding_parameters embed, int k, double *S)
int compute_entropy_ann_2d(double *x, int nx, int ny, int d, int p, int stride_x, int stride_y, 
                            int Theiler_x, int Theiler_y, int N_eff, int N_realizations, int k, int method, double *S)
{	register int jx, jy, p_new=p;
	int     N_eff_tot=0, N_real_tot=0; // effective values in 2d
	double *x_new, S_tmp, avg=0.0, var=0.0, x_new_std=0.0;
	double  *x_s; // shifted x
	int     do_shuffle=0;
	samp_param  sp_x = { .Theiler=Theiler_x,
                           .N_eff=N_eff,                // 2022-05-05: N_eff is propagated in N_eff_x
                           .N_real=N_realizations},     // 2022-05-12: N_realizations is propagated in N_realizations_x
                sp_y = { .Theiler=Theiler_y,
                           .N_eff=-1,                   // 2022-05-05: N_eff is propagated in N_eff_x (so not here)
                           .N_real=-1};
    gsl_permutation *perm_real_x, *perm_real_y;
	gsl_permutation *perm_pts_x,  *perm_pts_y;
//	FILE *fic=fopen("debug.txt", "wt");
	
#ifdef DEBUG
    debug_trace_2d("[compute_entropy_ann_2d] raw signal x", x, nx, ny);
    printf("[compute_entropy_ann_2d] : called with d=%d, p=%d (%d,%d), method=%d\n", d, p, stride_x, stride_y, method);
    printf("THEILER_2D = %d\n", THEILER_2D);
#endif

    *S = my_NAN; // default returned value
    if (Theiler_x<-2) do_shuffle=1; // 2022-05-19: if Theiler==-3 or -4, then we shuffle N_eff pts amongst N_eff_max
        
/*    printf("[compute_entropy_ann_2d] : before params checks in x (total N_eff_max=%d, total N_real_max=%d):\n", N_eff_tot, N_real_tot);
   	print_samp_param(sp_x);
   	printf("                           before params checks in y (total N_eff_max=%d, total N_real_max=%d):\n", N_eff_tot, N_real_tot);
   	print_samp_param(sp_y);
*/   	        
    if ((p<1))          return(printf("[compute_entropy_ann_2d] : embedding dimension p must be at least 1 !\n"));
    if ((stride_x<0))   return(printf("[compute_entropy_ann_2d] : stride in X cannot be negative !\n"));
    if ((stride_y<0))   return(printf("[compute_entropy_ann_2d] : stride in Y cannot be negative !\n"));
    if ((stride_x+stride_y)==0) 
                        return(printf("[compute_entropy_ann_2d] : stride in X and Y cannot both be zero !\n"));
    if ((k<1))          return(printf("[compute_entropy_ann_2d] : k must be at least 1 !\n"));
    if ((method<0) || (method>2))
                        return(printf("[compute_entropy_ann_2d] : method must be 0, 1 or 2 !\n"));
	
	// adapt embedding dimension according to method:
	if (method==0)  p_new = p;  // time-embedding (not increments)
	else            // increments (regular or averaged)
	{   p_new = 1;
	    p    += 1;  // convention: increments of order 1 (p) require 2 (p+1) points in order to be computed
	}
    
    // additional checks and auto-adjustments of parameters:
    N_real_tot = set_sampling_parameters_2d(nx, ny, p, stride_x, stride_y, &sp_x, &sp_y, "compute_entropy_ann_2d");   
    if (N_real_tot<1)   return(printf("[compute_entropy_ann_2d] : error in parameters; aborting !\n"));
    N_eff_tot  = sp_x.N_eff  * sp_y.N_eff;
    N_real_tot = sp_x.N_real * sp_y.N_real;
    
/*    printf("[compute_entropy_ann_2d] : after params checks in x (total N_eff=%d, total N_real=%d):\n", N_eff_tot, N_real_tot);
   	print_samp_param(sp_x);
   	printf("                           after params checks in y (total N_eff=%d, total N_real=%d):\n", N_eff_tot, N_real_tot);
   	print_samp_param(sp_y);
*/	
//return(-1);    
    if (N_eff_tot < 2*k)    
                        return(printf("[compute_entropy_ann_2d] : N_eff=%d is too small compared to k=%d)\n", N_eff_tot, k));

    x_new    = (double*)calloc(d*p_new*N_eff_tot, sizeof(double));

/*    fprintf(fic, "image %d x %d, max index %d\n", nx, ny, nx*ny);
    fprintf(fic, "stride_x: %d stride_y: %d\t ready to enter loop\n", stride_x, stride_y);
    fprintf(fic, "\talong x\n");
    fprintf(fic, "\t\tadapted Theiler : %d, (max : %d)\n", sp_x.Theiler,  sp_x.Theiler_max);
	fprintf(fic, "\t\tadapted N_eff   : %d, (max : %d)\n", sp_x.N_eff,    sp_x.N_eff_max); 
    fprintf(fic, "\t\tadapted N_real  : %d, (max : %d)\n", sp_x.N_real,   sp_x.N_real_max);
    fprintf(fic, "\n");
    fprintf(fic, "\talong y\n");
    fprintf(fic, "\t\tadapted Theiler : %d, (max : %d)\n", sp_y.Theiler,  sp_y.Theiler_max);
	fprintf(fic, "\t\tadapted N_eff   : %d, (max : %d)\n", sp_y.N_eff,    sp_y.N_eff_max); 
    fprintf(fic, "\t\tadapted N_real  : %d, (max : %d)\n", sp_y.N_real,   sp_y.N_real_max);
    fprintf(fic, "\n"); fflush(fic);
    fprintf(fic, "");
*/
    // "generate" independant windows:
    perm_real_x = create_unity_perm(sp_x.N_real_max); shuffle_perm(perm_real_x);
    perm_real_y = create_unity_perm(sp_y.N_real_max); shuffle_perm(perm_real_y);
    perm_pts_x  = create_unity_perm(sp_x.N_eff_max);
    perm_pts_y  = create_unity_perm(sp_y.N_eff_max);
//    fprintf(fic, "perms created\n"); fflush(fic);
    
    nb_errors=0; last_npts_eff=0;
    data_std=0;  data_std_std=0;        // 2022-03-11, for std of the increments
    for (jx=0; jx<sp_x.N_real; jx++)    // loop over independant windows in x
    for (jy=0; jy<sp_y.N_real; jy++)    // loop over independant windows in y
    {   if (do_shuffle>0)
        {   shuffle_perm(perm_pts_x);
            shuffle_perm(perm_pts_y);
        }
        // x_s = x + jy + jx*ny;                   // 2020-07-20 the orientation of the image is chosen to be the same as in Python
        x_s = x + stride_x*(p-1)*ny + stride_y*(p-1) + perm_real_y->data[jy] + perm_real_x->data[jx]*ny; // 2022-05-12 mixing solution
/*        fprintf(fic, "\tjx=%d/%d, jy=%d/%d.  ", jx, sp_x.N_real, jy, sp_y.N_real);
        fprintf(fic, "real starts at (%lu, %lu)", stride_x*(p-1)+ perm_real_x->data[jx], stride_y*(p-1) + perm_real_y->data[jy]);
        fprintf(fic, "\tso total shift is %lu\n", stride_x*(p-1)*ny + stride_y*(p-1) + perm_real_y->data[jy] + perm_real_x->data[jx]*ny);  
        fprintf(fic, "\t\t value %f  ", x_s[0]); fflush(fic);
        fprintf(fic, "\t\t max start would be at (%d, %d)", stride_x*(p-1)+ (sp_x.N_real_max-1), stride_y*(p-1) + (sp_y.N_real_max-1));
        fprintf(fic, "\tso total shift would be %d\n", (stride_x*(p-1)+ (sp_x.N_real_max-1))*ny + stride_y*(p-1) + (sp_y.N_real_max-1));
*/      
        if (method==0)      // regular embedding (not increments)
            Theiler_embed_2d(x_s, nx, ny, d, p, stride_x, stride_y, sp_x.Theiler, sp_y.Theiler, 
                                perm_pts_x->data, perm_pts_y->data, x_new, sp_x.N_eff, sp_y.N_eff);
        else if (method==1) // regular increments
            increments_2d   (x_s, nx, ny, d, p, stride_x, stride_y, sp_x.Theiler, sp_y.Theiler, 
                                perm_pts_x->data, perm_pts_y->data, x_new, sp_x.N_eff, sp_y.N_eff);
        else if (method==2) // averaged increments
            incr_avg_2d     (x_s, nx, ny, d, p, stride_x, stride_y, sp_x.Theiler, sp_y.Theiler, 
                                perm_pts_x->data, perm_pts_y->data, x_new, sp_x.N_eff, sp_y.N_eff);
//        fprintf(fic, "\tdone\n"); fflush(fic);
        // 2022-03-11: std of the signal, or of the increments, well of the data under operation:
        // 2022-05-12: note that this works OK only if d==1 and if p_new==1 ... 
        x_new_std     = gsl_stats_sd(x_new, 1, N_eff_tot); // GSL: unbiased estimator
//        x_new_std     = gsl_stats_mean(x_new, 1, sp_x.N_eff*sp_y.N_eff); // 2022-05-19 for tests only
        data_std     += x_new_std;
        data_std_std += x_new_std*x_new_std;

        
#ifdef DEBUG
        for (int l=0; l<d*p_new; l++)
            debug_trace_2d("sub-matrix x_new", x_new+l*npts_new, sp_x.N_eff, sp_y.N_eff); // first dimension = 1 image
        debug_trace_2d("full x_new", x_new, npts_new, 1);
        printf("\n");
#endif
        if (USE_PTHREAD>0) 
             S_tmp = compute_entropy_nd_ann_threads(x_new, N_eff_tot, d*p_new, k, get_cores_number(GET_CORES_SELECTED));
        else S_tmp = compute_entropy_nd_ann        (x_new, N_eff_tot, d*p_new, k);
        
        avg  += S_tmp;
        var  += S_tmp*S_tmp;
        nb_errors += nb_errors_local; // each call to "compute_entropy_nd_ann" gives a new value of nb_errors_local
        last_npts_eff += last_npts_eff_local; // each call to "compute_entropy_nd_ann" uses an effective nb of points
    }
    avg       /= N_real_tot;
    data_std  /= N_real_tot;
    if (N_real_tot>1)
    {   var /= N_real_tot-1;
        var -= avg*avg * N_real_tot/(N_real_tot-1);                     // unbiased estimator;
        data_std_std /= (N_real_tot-1);    
        data_std_std -= data_std*data_std * N_real_tot/(N_real_tot-1);  // unbiased estimator
        data_std_std = sqrt(data_std_std);
    }
    else 
    {   var          = 0.;
        data_std_std = 0.;
    }
    
    *S = avg;
    last_std = sqrt(var);
    
    nb_errors_total += nb_errors;
    last_samp=sp_x;
    samp_2d.last_Theiler_x = sp_x.Theiler; 
    samp_2d.last_Theiler_y = sp_y.Theiler;
    
//    fclose(fic);
    free(x_new);
    free_perm(perm_real_x); free_perm(perm_real_y);
	free_perm(perm_pts_x);  free_perm(perm_pts_y);
	return(nb_errors);
} /* end of function "compute_entropy_2d" *************************************/



