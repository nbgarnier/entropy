/*
 *  entropy_others.c : for ApEn and SampEn
 *  
 *
 *  Created by Nicolas Garnier on 2013/04/19.
 *  Copyright 2013 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2013-04-19 : fork from entropy_symb.c: 2 new functions for ApEn and SampEn
 *  2020-03-04 : new single function, returning both ApEn and SampEn
 *  2023-12-06 : new function using masks and advanced samplings
 */

#include <math.h>	            // for fabs
#include <string.h>             // for memcpy

#include "verbosity.h"
#include "samplings.h"
#include "mask.h"

#include "library_commons.h"  
#include "library_matlab.h"     // compilation for Matlab
#include "math_tools.h"         // for Quicksort et al

#include "entropy_others.h"


#define noDEBUG		// for debug information, replace "noDEBUG" by "DEBUG"
#define LOOK 167	// for debug also (of which point(s) will we save the data ?)


// global variables for efficiency :
double *Complexity_Kernel     = NULL;
int    *Complexity_Kernel_int = NULL;

/****************************************************************************************/
/* computes Approximate entropy (Pincus 1991)                                           */
/*																			            */
/* x       contains all the data, which is of size npts in time						    */
/* npts    is the number of points in time										        */
/* m       indicates how many points to take in time (in the past) (embedding)          */
/* stride  is the time lag between 2 consecutive points to be considered in time		*/
/* r       is the radius of the balls for counting (resolution)                         */
/*																			            */
/* 2013-04-23 : first version, no stride                                                */
/*          inspired by http://www.codeproject.com/Articles/27030/Approximate-and-Sample-Entropies-Complexity-Metric */
/*          but corrected                                                               */
/* 2020-02-04 : remark: this is not causal!                                             */
/****************************************************************************************/
double compute_ApEn_old(double* data, int m, double r, int npts)
{   register int i,j,k;
    int Cm = 0, Cm1 = 0;
    double sum = 0.0;
//  double  ApEn[10], SampEn[10];
    int is_close;
        
    for (i=0; i<(npts-m); i++) 
    {   Cm  = 0;
        Cm1 = 0;
        for (j=0; j<(npts-m); j++) 
        {   
            is_close = 1;
            k=0;
            while ( (k<m) && (is_close==1) ) 
            {  if (fabs(data[i+k] - data[j+k]) > r) 
                    is_close = 0;
                k++;
            }
            
            if (is_close==1) 
            {   Cm++; // one more neighbor at rank m
                    
                // then check at rank (m+1):
                k = m;
                if (fabs(data[i+k] - data[j+k])<= r)  
                 Cm1++;
            }
        }

        // all Cm and Cm1 are stricly positive by construction (cf j=i gives a match)
        sum += log((double)Cm) - log((double)Cm1);                
    }
    
 /*   printf("measure ApEn : %f (old)\n", sum / (double)(npts - m));
    
    // test :
     compute_complexity(data, m+1, r, npts, KERNEL_GAUSSIAN, ApEn, SampEn);
    printf("new : ");
    for (i=0; i<m; i++)
        printf("%f (m=%d)\n", ApEn[i], i);
*/
    return (sum / (double)(npts - m));
}


/***************************************************************************************/
/* computes Sample entropy (Richman - Moorman 2002)                                    */
/*																			*/
/* x       contains all the data, which is of size npts in time						*/
/* npts    is the number of points in time										*/
/* m       indicates how many points to consider in time (embedding)                   */
/* stride  is the time lag between 2 consecutive points to be considered in time		*/
/* r       is the radius of the balls for counting (resolution)                        */
/*																			*/
/* 2013-04-23 : first version, no stride                                               */
/***************************************************************************************/
double compute_SampEn_old(double* data, int m, double r, int npts)
{   register int i,j,k;
    int Cm = 0, Cm1 = 0;
    int is_close;
    double SampEn=0.0;
    
    for (i=0; i< (npts-m); i++) 
    {
        for (j=i+1; j< (npts-m); j++) 
        {      
            is_close = 1;
            k=0;
            while ( (k<m) && (is_close==1) ) 
            {   if (fabs(data[i+k] - data[j+k]) > r) 
                {    is_close = 0;}
                k++;
            }

            if (is_close==1) 
            {   Cm++; // one more neighbor at rank m
                // then check at rank m+1:
                k = m;
                if (fabs(data[i+k] - data[j+k])<= r)    Cm1++;
            }
        }
    }
//    printf("Cm : %d, Cm1 : %d\n", Cm, Cm1);
    if ( (Cm>0) && (Cm1>0) )    SampEn = log((double)Cm) - log((double)Cm1);

    return(SampEn);
}


/**************************************************************************************/
/* pre-compute quantities for Approximate and Sample entropy                          */
/*														                              */
/* x       contains all the data, which is of size npts in time			              */
/* npts    is the number of points in time										      */
/* m       indicates how many points to consider in time (embedding)                  */
/*         ApEn and SampEn will be of order (m-1) : H_(m)-H_(m-1)                     */
/* stride  is the time lag between 2 consecutive points to be considered along time	  */
/* r       is the radius of the balls for counting (= the resolution)                 */
/*																			          */
/* returns an array with A_i and B_i                                                  */
/*                                                                                    */
/* 2014-03-30 : first version, no stride                                              */
/* 2020-03-04 : now with stride, but no Theiler correction!!!                         */
/* 2023-12-06 : renamed with "_embed" suffix                                         */
/**************************************************************************************/
double prepare_kernel_brickwall_embed(double* data, int npts, int m, int stride, double r)
{   register int i,j,k;
    int is_close;
        
    Complexity_Kernel = (double *)calloc(npts*m, sizeof(double));
        
    for (i=(m-1)*stride; i<npts; i++) 
    {   for (j=i+1; j<npts; j++) // other points, not counting points twice (pairs are examinated only once)
        {   
            is_close = 1;
            k=0;
            while ( (k<m) && (is_close==1) )          // loop on embedding dimensions
            {   if (fabs(data[i-k*stride] - data[j-k*stride]) > r) 
                    is_close = 0;       // not good
                else    // the point is close at order k
                {
                    Complexity_Kernel[k*npts + i] ++; // 1 more neighbors to point i
                    Complexity_Kernel[k*npts + j] ++; // 1 more neighbors to point j                    
                }
                k++;
            }
        }
    }
    
    // normalisation: (useless for ApEn or SampEn), we return the normalisatino instead
//    for (i=(m-1)*stride; i<npts; i++) 
//    for (k=0; k<m; k++) Complexity_Kernel[k*npts + i] /= (npts-(m-1)*stride);

    return(npts-(m-1)*stride);
}


/*************************************************************************************/
/* pre-compute quantities for Approximate and Sample entropy                         */
/*																			         */
/* x       contains all the data, which is of size npts in time						 */
/* npts    is the number of points in time										     */
/* m       indicates how many points to consider in time (embedding)                 */
/*         ApEn and SampEn will be of order (m-1) : H_(m)-H_(m-1)                    */
/* stride  is the time lag between 2 consecutive points to be considered in time     */
/* r       is the radius of the balls for counting (resolution)                      */
/*																			         */
/* returns an array with A_i and B_i                                                 */
/*                                                                                   */
/* 2014-03-30 : first version, no stride                                             */
/* 2020-03-04 : huge rewritting                                                      */
/* 2020-03-04 : now with stride, but no Theiler correction!!!                        */
/* 2023-12-06 : renamed with "_embed" suffix                                         */
/*************************************************************************************/
double prepare_kernel_Gaussian_embed(double* data, int npts, int m, int stride, double r)
{   register int i,j,k;
    double a, d, norm=2*r*r;
        
    Complexity_Kernel = (double *)calloc(npts*m, sizeof(double));
//    a                 = (double *)calloc(npts,   sizeof(double));
//    b                 = (double *)calloc(npts,   sizeof(double));

    for (i=(m-1)*stride; i<npts; i++) 
    {   for (j=i+1; j<npts; j++)    // using symetry of pairs, and not allowing (i==j)
        {   a=1.0;
            for (k=0; k<m; k++) 
            {   d = data[i-k*stride]-data[j-k*stride];
                a *= exp(-d*d/norm);
                Complexity_Kernel[k*npts + i] += a;
                Complexity_Kernel[k*npts + j] += a;
            }
        }
    }
//    free(a);
//    free(b);

    return(1.0);  // 2020-03-04: normalisation to be computed...
}



void free_complexity_kernel(void)
{   free(Complexity_Kernel);
    return; 
}



/**************************************************************************************/
/* Compute Approximate and Sample entropy                                             */
/*																			          */
/* this uses the function "prepare_kernel_brickwall"                                  */
/*                                                                                    */
/* x       contains all the data, which is of size npts in time						  */
/* npts    is the number of points in time										      */
/* m       indicates how many points to consider in time (embedding)                  */
/*         ApEn and SampEn will be of order (m) : H_(m+1)-H_(m)                       */
/* stride  is the time lag between 2 consecutive points to be considered in time	  */
/* r       is the radius of the balls for counting (resolution)                       */
/* kernel_type indicates either :                                                     */
/*          ==0: regular (brickwall) kernel with max norm                             */
/*          ==1: Gaussian kernel                                                      */
/*																			          */
/*                                                                                    */
/* 2014-03-30 : first version, no stride                                              */
/* 2020-03-04 : new version, with stride, but no Theiler correction                   */
/* 2023-12-06 : renamed functions "prepare_kernel_XXX" to "prepare_kernel_XXX_embed"  */
/**************************************************************************************/
int compute_complexity(double* data, int npts, int m, int stride, double r, int kernel_type, double *ApEn, double *SampEn)
{   register int i,k;
    double *sum, *sum_log, norm=1.0;
   
    if ( (m<0) || (r==0) ) return(-1);
    
    if (kernel_type==KERNEL_BRICKWALL)      
    {   norm = prepare_kernel_brickwall_embed(data, npts, m+1, stride, r);
        // norm may not have been applied inside the function constructing the kernel,
        // in that case, it is returned by the function
    }
    else if (kernel_type==KERNEL_GAUSSIAN)  
    {   norm = prepare_kernel_Gaussian_embed(data, npts, m+1, stride, r);
        // not correct, but this does not matter for ApEn or SampEn (only for entropy)
    }
    else return(-1);

    sum     = (double*)calloc((m+1),sizeof(double));
    sum_log = (double*)calloc((m+1),sizeof(double));        
        
    for (i=(m+1-1)*stride; i<npts; i++) // nb of pts decided by max embedding dimension
    {   for (k=0; k<m+1; k++)
        {   sum    [k] += (double)Complexity_Kernel[k*npts + i];            // for SampEn
            sum_log[k] += log((double)Complexity_Kernel[k*npts + i] + 1.0); // for ApEn // we add the 'central point'
        } 
    }

    // for k=0, we compute the entropy (not the entropy rate!):
    ApEn  [0] = - (sum_log[0] )/(npts-(m+1-1)*stride) + log(norm);
    SampEn[0] = - log(sum[0]/norm) + log(npts-(m+1-1)*stride);
//    printf("ApEn(1)   = H -log(2epsilon) :  %f = %f - %f = %f\n", ApEn[0],   -1.4, log(2*r),  -1.4-log(2*r) );
//    printf("SampEn(1) = H -log(2epsilon) :  %f = %f - %f = %f\n", SampEn[0], -(1.4+(log(2)-1)/2), log(2*r), -(1.4+(log(2)-1)/2)-log(2*r) );
    
    // for larger k values, these will be the entropy rates:
    for (k=1; k<m+1; k++)
    {   ApEn[k]   = - (sum_log[k] - sum_log[k-1])/(npts-(m+1-1)*stride);
        SampEn[k] = - log(sum[k]) + log(sum[k-1]);
    }
    
    // free the memory:
    free(sum); free(sum_log);
    free_complexity_kernel();
  
    return(0); // no problem
}


void *alloc_complexity_kernel(int npts, int m)
{   Complexity_Kernel = (double *)calloc(npts*m, sizeof(double));
    return(Complexity_Kernel);
}

void reset_array_d(double *x, int n)
{   register int i;
    for (i=0; i<n; i++) x[i]=0.0;
}

/************************************************************************************/
/* pre-compute quantities for Approximate and Sample entropy                        */
/*														                            */
/* x       contains all the data, which is of size npts in time			            */
/* npts    is the number of points in time										    */
/* m       indicates the dimensionality of x (which is pre-time-embedded)           */
/* r       is the radius of the balls for counting (= the resolution)               */
/*																			        */
/* note: the global variable "Complexity_Kernel" must be allocated!                 */
/*																			        */
/* returns the nb of points used                                                    */
/*                                                                                  */
/* 2014-03-30 : first version, no stride                                            */
/* 2020-03-04 : now with stride, but no Theiler correction!!!                       */
/* 2023-12-06 : forked from "prepare_kernel_brickwall"                              */
/************************************************************************************/
double apply_kernel_brickwall_nd(double* data, int npts, int m, double r)
{   register int i,j,k;
    int is_close;
        
//    Complexity_Kernel = (double *)calloc(npts*m, sizeof(double));
        
    for (i=0; i<npts; i++) 
    {   for (j=i+1; j<npts; j++) // other points, not counting points twice (pairs are examinated only once)
        {   
            is_close = 1;
            k=0;
            while ( (k<m) && (is_close==1) )          // loop on embedding dimensions
            {   if (fabs(data[i+k*npts] - data[j+k*npts]) > r) 
                    is_close = 0;       // not good
                else    // the point is close at order k
                {   Complexity_Kernel[k*npts + i] ++; // 1 more neighbors to point i
                    Complexity_Kernel[k*npts + j] ++; // 1 more neighbors to point j                    
                }
                k++;
            }
        }
    }
    
    // normalisation: (useless for ApEn or SampEn), we return the normalisatino instead
//    for (i=0; i<npts; i++) 
//    for (k=0; k<m; k++) Complexity_Kernel[k*npts + i] /= (npts);

    return(npts);
}



/************************************************************************************/
/* pre-compute quantities for Approximate and Sample entropy                        */
/*																			        */
/* x       contains all the data, which is of size npts in time			            */
/* npts    is the number of points in time										    */
/* m       indicates the dimensionality of x (which is pre-time-embedded)           */
/* r       is the radius of the balls for counting (= the resolution)               */
/*																			        */
/* note: the global variable "Complexity_Kernel" must be allocated!                 */
/*																			        */
/* returns the nb of points used                                                    */
/*                                                                                  */
/* 2014-03-30 : first version, no stride                                            */
/* 2020-03-04 : huge rewritting                                                     */
/* 2020-03-04 : now with stride, but no Theiler correction!!!                       */
/* 2023-12-06 : forked from "prepare_kernel_brickwall"                              */
/*************************************************************************************/
double apply_kernel_Gaussian_nd(double* data, int npts, int m, double r)
{   register int i,j,k;
    double a, d, norm=2*r*r;
        
//    Complexity_Kernel = (double *)calloc(npts*m, sizeof(double));

    for (i=0; i<npts; i++) 
    {   for (j=i+1; j<npts; j++)    // using symetry of pairs, and not allowing (i==j)
        {   a=1.0;
            for (k=0; k<m; k++) 
            {   d = data[i+k*npts]-data[j+k*npts];
                a *= exp(-d*d/norm);
                Complexity_Kernel[k*npts + i] += a;
                Complexity_Kernel[k*npts + j] += a;
            }
        }
    }

    return(1.0);  // 2020-03-04: normalisation to be computed...
}



/**************************************************************************************/
/* Compute Approximate and Sample entropy using masks and advanced samplings          */
/*																			          */
/* this uses the function "prepare_kernel_brickwall"                                  */
/*                                                                                    */
/* x       contains all the data, which is of size npts in time						  */
/* npts    is the number of points in time										      */
/* m       indicates how many points to consider in time (embedding)                  */
/*         ApEn and SampEn will be of order (m) : H_(m+1)-H_(m)                       */
/* stride  is the time lag between 2 consecutive points to be considered in time	  */
/* r       is the radius of the balls for counting (resolution)                       */
/* kernel_type indicates either :                                                     */
/*          ==0: regular (brickwall) kernel with max norm                             */
/*          ==1: Gaussian kernel                                                      */
/*																			          */
/*                                                                                    */
/* 2014-03-30 : first version, no stride                                              */
/* 2020-03-04 : new version, with stride, but no Theiler correction                   */
/* 2023-12-06 : forked from "compute_complexity"                                      */
/* 2023-12-08 : tested OK (see python script)                                         */
/**************************************************************************************/
int compute_complexity_mask(double* data, char *mask, int npts, int m, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, 
                            double r, int kernel_type, double *ApEn, double *SampEn)
{   register int i,j,k;
    double  *x_new, sum[m+1], sum_log[m+1], norm=1.0;
    double  ae[m+1], se[m+1], avg_ae[m+1], avg_se[m+1], var_ae=0.0, var_se=0.0;
    int     N_real_max=0, npts_good;
    int     *ind_epoch=NULL;
    size_t  *ind_shuffled=NULL;
    samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations, .type=-1};
    gsl_permutation *perm_pts;
    char message[128];
    
    *ApEn=my_NAN;  *SampEn=my_NAN;

    if (m<0)        return(print_error("compute_complexity_mask", "m cannot be negative"));
    if (r==0)       return(print_error("compute_complexity_mask", "r must be strictly positive"));
    if (stride<1)   return(print_error("compute_complexity_mask", "stride must be at least 1"));
    
    N_real_max = set_sampling_parameters_mask(mask, npts, m+1, stride, 0, &sp, "compute_complexity_mask");
    if (N_real_max<1) return(print_error("compute_complexity_mask", "not enough points, aborting"));
    
    x_new   = (double*)calloc((m+1)*sp.N_eff, sizeof(double));
    alloc_complexity_kernel(sp.N_eff, m+1);
    reset_array_d(avg_ae, m+1);     // maybe not necessary
    reset_array_d(avg_se, m+1);
    
    nb_errors=0; last_npts_eff=0;
    for (j=0; j<sp.N_real; j++)         // loop over independant windows (independant !!! cf Theiler !!!)
    {  
        npts_good = analyze_mask_for_sampling(mask+j, npts-j, m+1, stride, 0, sp.Theiler, 1, &ind_epoch); 
        if (npts_good<sp.N_eff) 
        {   sprintf(message, "npts_good = %d < N_eff =%d", npts_good, sp.N_eff);
            return(print_error("compute_complexity_mask", message));
        }   
        perm_pts  = create_unity_perm(npts_good); shuffle_perm(perm_pts); // for random sampling
        ind_shuffled = (size_t*)calloc(npts_good, sizeof(size_t));
        for (i=0; i<npts_good; i++) ind_shuffled[i] = ind_epoch[perm_pts->data[i]]; // OK

        Theiler_embed_mask(data+j, npts, 1, m+1, stride, ind_shuffled, x_new, sp.N_eff);
        
        // now real math:        
        if (kernel_type==KERNEL_BRICKWALL)      norm=apply_kernel_brickwall_nd(x_new, sp.N_eff, m+1, r);
        else if (kernel_type==KERNEL_GAUSSIAN)  norm=apply_kernel_Gaussian_nd (x_new, sp.N_eff, m+1, r);
        // norm may not have been applied inside the function constructing the kernel,
        // in that case, it is returned by the function
        // not correct, but this does not matter for ApEn or SampEn (only for entropy, i.e., for m=0)
// to adapt:       
        reset_array_d(sum,     m+1);
        reset_array_d(sum_log, m+1);        
        
        for (i=0; i<sp.N_eff; i++) 
        {   for (k=0; k<m+1; k++)
            {   sum    [k] += (double)Complexity_Kernel[k*sp.N_eff + i];            // for SampEn
                sum_log[k] += log((double)Complexity_Kernel[k*sp.N_eff + i] + 1.0); // for ApEn // we add the 'central point'
            } 
        }
        // for k=0, we compute the entropy (not the entropy rate!):
        ae[0] = - (sum_log[0] )/sp.N_eff + log(norm);
        se[0] = - log(sum[0]/norm) + log(sp.N_eff);
//    printf("ApEn(1)   = H -log(2epsilon) :  %f = %f - %f = %f\n", ApEn[0],   -1.4, log(2*r),  -1.4-log(2*r) );
//    printf("SampEn(1) = H -log(2epsilon) :  %f = %f - %f = %f\n", SampEn[0], -(1.4+(log(2)-1)/2), log(2*r), -(1.4+(log(2)-1)/2)-log(2*r) );
        // for larger k values, these will be the entropy rates:
        for (k=1; k<m+1; k++)
        {   ae[k] = - (sum_log[k] - sum_log[k-1])/sp.N_eff;
            se[k] = - log(sum[k]) + log(sum[k-1]);
        }

        for (k=0; k<m+1; k++)
        {   avg_ae[k]  += ae[k];
            avg_se[k]  += se[k];
        }
        var_ae += ae[m]*ae[m];
        var_se += se[m]*se[m];

        free(ind_epoch);
        free(ind_shuffled);
        free_perm(perm_pts);        
    }
    for (k=0; k<m+1; k++) { avg_ae[k] /= sp.N_real;     avg_se[k] /= sp.N_real;  }
    var_ae /= sp.N_real;        var_ae -= avg_ae[m]*avg_ae[m];
    var_se /= sp.N_real;        var_se -= avg_se[m]*avg_se[m];

    for (k=0; k<m+1; k++) { ApEn[k] = avg_ae[k];        SampEn[k] = avg_se[k];    }
    last_std = sqrt(var_ae);    last_std2 = sqrt(var_se);
    last_samp=sp;
        
    // free the memory:
    free_complexity_kernel();
  
    free(x_new);
    return(0); // no problem
} // end oof function "compute_complexity_mask"
