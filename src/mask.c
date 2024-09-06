/*
 *  mask.c
 *  
 *  Created by Nicolas Garnier on 2013/06/20.
 *  Copyright 2013 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2013-06-20: new source file containing all functions related with masking and epochs
 *  2020-02-23: new function "combine_masks"
 *  2023-12-01: several improvments and tests 
 *  2023-12-04: moved deprecated functions to "mask_old.c"
 */

#include <math.h>               // for fabs and nan operations */
#include <string.h>

#include "samplings.h"          // for sampling functions
#include "verbosity.h"          // for verbosity
#include "library_matlab.h"     // compilation for Matlab
#include "math_tools.h"

#define noDEBUG		// for debug information, replace "noDEBUG" by "DEBUG"
#define LOOK 167	// for debug also (of which point(s) will we save the data ?)


/********************************************************************************/
/* to create a mask with 0 if real number, and 1 if NaN                         */
/*                                                                              */
/* x    is the data (ordered as x_0(t=0), x_0(t=1), ..., x_(m-1)(npts-1)        */
/* npts is the nb of points in time                                             */
/* m    is the dimensionality                                                   */
/*                                                                              */
/* mask is the returned mask (of size npts): should be previously allocated !   */
/*                                                                              */
/* 2020-03-01 : first version, untested                                         */
/********************************************************************************/
int NaN_mask(double *x, int npts, int m, char *mask)
{   register int i,j=0;

    for (i=0; i<npts; i++) mask[i]=0;
    for (i=0; i<npts*m; i++)
        if (isnan(x[i])) 
        {   mask[i%npts]|=1;
            j++;
        }
    return(j);        
}



/********************************************************************************/
/* to create a mask with 1 if real number, and 0 if NaN or INF                  */
/*                                                                              */
/* x    is the data (hould be order as x_0(t=0), x_0(t=1), ..., x_(m-1)(npts-1) */
/* npts is the nb of points in time                                             */
/* m    is the dimensionality                                                   */
/*                                                                              */
/* mask is the returned mask (of size npts): should be previously allocated !   */
/*                                                                              */
/* 2020-03-01 : first version, untested                                         */
/********************************************************************************/
int finite_mask(double *x, int npts, int m, char *mask)
{   register int i,j=0;

    for (i=0; i<npts; i++) mask[i]=1;
    for (i=0; i<npts*m; i++)
        if (!isfinite(x[i]))
        {   mask[i%npts]=0;
            j++;
        }
    return(j);        
}



/********************************************************************************/
/* to combine two masks into a single one, using AND logic                      */
/*                                                                              */
/* mask_1    contains the first mask                                            */
/* mask_2    contains the second mask                                           */
/* npts      is the size of mask (nb of points in time)                         */
/*                                                                              */
/* mask_out  is the returned mask (must be previously allocated)                */
/*                                                                              */
/* 2020-02-23 : new function (to compute entropy_rate)                          */
/* 2023-11-29 : replaced | by &*/
/********************************************************************************/
void combine_masks(char *mask_1, char *mask_2, int npts, char *mask_out)
{   register int i;
    
    for (i=0; i<npts; i++) mask_out[i] = mask_1[i] & mask_2[i]; // 2020-02-28: "|" or "+"
    // 2023-11-29: replaced | by &
}



/********************************************************************************/
/* to "project" or "retain" points of a signal x according to a mask            */
/*                                                                              */
/* x    is the data (should be order as x_0(t=0), x_0(t=1), ..., x_(m-1)(npts-1)*/
/* npts is the nb of points in time                                             */
/* m    is the dimensionality                                                   */
/* mask is the mask                                                             */
/*                                                                              */
/* y    is the output (of size (npts_new,m)   must be pre-allocated!            */
/* npts_new is the returned value                                               */
/*                                                                              */
/* mask is the returned mask (of size npts): should be previously allocated !   */
/*                                                                              */
/* 2020-03-01 : first version, untested                                         */
/********************************************************************************/
int retain_from_mask(double *x, int npts, int m, char *mask, double *y)
{   register int i,j,d;
    int npts_new=0;

    for (i=0; i<npts; i++) if (mask[i]>0) npts_new++;
    
    j=0;
    for (i=0; i<npts; i++)
    {   if (mask[i]>0)
        {   for (d=0; d<m; d++) // other dimensions
               y[j + d*npts_new] = x[i + d*npts];
            j++;
        }
    }
    
    return(npts_new);
}

/********************************************************************************/
/* search within a mask for sets of indices to work on                          */
/*                                                                              */
/* mask               contains the mask.  should be 1-dimensional!!!            */
/* npts               the size of mask (nb of points in time)                   */
/* p                  the desired embedding dimension                           */
/* stride             the time lag for embedding                                */
/* lag                the time lag for TE  (set it to 0 if useless)             */
/* Theiler            the Theiler stride between 2 embedded points to examine   */
/* do_return_indices  if 0, no indices are returned; if 1, indices are returned */
/* ind_epoch (returned value) will contain the indices of beginning of sets     */
/*                                                                              */
/* main returned value (int) is the number of relevant sets found               */
/*                                                                              */
/* 2020-02-27 : FYI, a set is a point where causal embedding can be performed   */
/* 2022-05-28 : forked from "analyse_mask_Theiler_optimized"                    */
/********************************************************************************/
int analyze_mask_for_sampling(char *mask, int npts, int p, int stride, int lag, int Theiler, int do_return_indices, int **ind_epoch)
{   register int i=0, j=0, n_sets=0;
    int     *ind_epoch_tmp=NULL; // times of beginning of epochs
    char    is_good_set;
    
    if (do_return_indices) ind_epoch_tmp = (int*)calloc((int)(npts), sizeof(int)); // 2016-05-17, allocate maximally
    
    n_sets = 0; // number of the set under study, or the next one
    for (i=(p-1)*stride; i<(npts-lag); i+=Theiler) // scan all available begining points for embedding domains
                                            // Theiler protocol eventually applied
    {
        is_good_set = 1;
        j = 0;
        while ( (j<p) && is_good_set )          // check causal embedding
        {   is_good_set = (mask[i-j*stride]>0) ? 1 : 0;
            j++;
        }   
        if (is_good_set) is_good_set = (mask[i+lag]>0) ? 1 : 0;  // check future lag
        
        if (is_good_set>0)
        {   if (do_return_indices) ind_epoch_tmp[n_sets] = i;
            n_sets++;
        }
    }
    
    if ( (do_return_indices>0) && (n_sets>0) )
    {   *ind_epoch = (int*)calloc(n_sets, sizeof(int)); // allocate exactly
        memcpy(*ind_epoch, ind_epoch_tmp, n_sets*sizeof(int));
    } // 2022-05-28: note: use realloc instead
    
    if (do_return_indices) free(ind_epoch_tmp);
    return(n_sets);
} /* end of function "analyze_mask_for_sampling" */



/********************************************************************************/
/* search minimal N_eff_max allowed by a given mask and sampling parameters     */
/*                                                                              */
/* mask            contains the indices of good epochs                          */
/* npts            the size of data / mask                                      */
/* p, stride, lag: sampling parameters                                          */
/* tau_Theiler     Theiler sampling scale.                                      */
/* N_real          nb of realizations                                           */
/*                                                                              */
/* returned value N_eff_max (int) = max available number of effective points    */
/*                                                                              */
/* 2023-12-01 : new function to simplify code reading                           */
/*              a quicksort may be a good idea to get the longest last          */
/********************************************************************************/
int get_N_eff_from_mask(char *mask, int npts, int p, int stride, int lag, int tau_Theiler, int N_real)
{   register int j, N_eff;
    int *N_eff_max  = (int*)calloc(N_real, sizeof(int));    // we'll search for N_real realization, not more.
                                                    // 2023-11-29: this could also be "stride" realizations, but we save time
    
    for (j=0; j<N_real; j++)
    {   N_eff_max[j] = analyze_mask_for_sampling(mask+j, npts-j, p, stride, lag, tau_Theiler, 0, NULL);
    } 
    N_eff       = find_min_int(N_eff_max, N_real);
    
    if (lib_verbosity>3)    
    {   printf("[get_N_eff_from_mask] for scale %d and Theiler %d\n", stride, tau_Theiler);
        printf("\t N_eff_max = [ ");
        for (j=0; j<N_real; j++) printf("%d  ", N_eff_max[j]);
        printf("] ");
    }
    if (lib_verbosity>2) 
        printf("[get_N_eff_from_mask]\t=> N_eff_max = %d\n", N_eff);
    
    free(N_eff_max);
    return(N_eff);
}



/****************************************************************************************/
/* computes optimal parameters, depending on user's requirements                        */
/*                                                                                      */
/* npts           : nb of points in dataset                                             */ 
/* p              : embedding dimension                                                 */ 
/* stride         : analysis scale                                                      */
/* tau_Theiler    : Theiler sampling,                                                   */
/*                  ==0  requests NO Theiler prescription                               */
/*                          (in that case, N_real is set to 1)                          */
/*                  ==-1 requests autosetting (=stride)  + uniform sampling (legacy)    */
/*                  ==-2 is not supported for masks => corresponds to 3                 */
/*                  ==-3 requests autosetting (=stride)  + random sampling              */
/*                  ==-4 requests autosetting (possibly <stride!) + random sampling     */
/*                          (this case, imposed N_eff may lead to reducing Theiler)     */
/*                                                                                      */
/* N_eff          : nb of effective points to use, value ==-1 requests auto-detection   */ 
/*                                                                                      */
/* N_realizations : nb of realizations, value ==-1 requests legacy auto-detection       */
/*                                      value ==-2 requests optimized auto-detection    */
/* function_name  : name of the calling function                                        */
/*                                                                                      */
/* on success, returns N_real_max, the number of available possible realizations        */
/* on failure, returns -1                                                               */
/*                                                                                      */
/* 2022-05-17: forked from "set_sampling_parameters"                                    */
/* 2023-11-29: the optionss from tau_Theiler for auto-adjust should be re-labeled       */
/* 2023-12-01: full rewriting. Now the function should be called with sp->type=0 or -1  */
/*              in order to operate properly when invoked from a compute_XXX function   */
/* 2023-12-02: now Theiler type 4 also allows for an increase of tau_Theiler            */
/****************************************************************************************/
int set_sampling_parameters_mask(char *mask, int npts, int p, int stride, int lag, samp_param *sp, char *func_name)
{   int do_auto_Neff=0;
	int sampling_ratio=1;
	double ratio;
    
    if (sp->type<0)                                 // 2023-12-01: now the function can be called recursively
    {   if (sp->Theiler>0)          sp->type=0;     // tau_Theiler is imposed (this is indeed a viable option)
        else if (sp->Theiler==-4)   sp->type=4;     // 4: we may reduce Theiler to have the required N_eff
        else if (sp->Theiler==-3)   sp->type=3;     // for tau_Theiler=tau and random sampling
        else if (sp->Theiler==-2)   return(-1);     // for a smarter adjustment (spanning all the dataset)
        else if (sp->Theiler==-1)   sp->type=1;     // for tau_Theiler=tau and uniform sampling (legacy)
        else if (sp->Theiler<-4)    return(-1);     // bad call
    }
    
    if (sp->N_eff<1)            do_auto_Neff=1;     // for (legacy)
    
    // we first study the requirements on N_real:
    if (sp->N_real<1)           sp->N_real=1;       // N_real should usualy be specified! new default value for automatic
    sp->N_real_max = sp->N_real;                    // default return value
    
    if (sp->type>0)             sp->Theiler=stride; // if Theiler scale is not given, we start by auto-adapting
    sp->Theiler_max = sp->Theiler;                  // the max has (for now) no meaning with masks
    
    if (sp->Theiler==0)                             // very old method introduced on 2011-11-14 : no Theiler-correction !
    {	sp->type        = 0;
        sp->Theiler     = 1;                        // to be able to use imposed N_eff and N_real
        sp->Theiler_max = 1;                        // 2022-12-14
        sp->N_real_max  = 1;                        // legacy setting when no Theiler prescription
        sp->N_real      = 1;                        // 2022-12-07: may not be necessary
        sp->N_eff_max   = get_N_eff_from_mask(mask, npts, p, stride, lag, sp->Theiler, sp->N_real);
        if (sp->N_eff<1) sp->N_eff = sp->N_eff_max;
    }
    else
    if (sp->type==0)                                // Theiler scale is imposed : it will not be adapted
    {   sp->Theiler_max = sp->Theiler;              // 2022-12-14
        if (sp->N_real<1) sp->N_real=stride;
        sp->N_real_max  = sp->N_real;
        
//        printf("Theiler type %d, for scale %d and imposed Theiler %d\n", sp->type, stride, sp->Theiler);
        sp->N_eff_max   = get_N_eff_from_mask(mask, npts, p, stride, lag, sp->Theiler, sp->N_real);
        if (sp->N_eff<1) sp->N_eff = sp->N_eff_max; // N_eff automatic
    }
    else
    if (sp->type==1)                                // automatic Theiler style 1 (legacy, 2012-05-03)
    {   sp->Theiler     = stride;
        sp->Theiler_max = sp->Theiler;              // 2022-12-14
        sp->N_real      = stride;                   // 2022-12-07: this crucial line was missing!
        sp->N_real_max  = sp->N_real;
//        printf("Theiler type %d, for scale %d and legacy Theiler %d\n", sp->type, stride, sp->Theiler);
        sp->N_eff_max   = get_N_eff_from_mask(mask, npts, p, stride, lag, sp->Theiler, sp->N_real);
        sp->N_eff       = sp->N_eff_max;
    }
    else
    if (sp->type==3)                                // automatic Theiler style 3
    {   sp->Theiler     = stride;
        sp->Theiler_max = sp->Theiler;              // 2022-12-14, tau_Theiler is fixed (not adapted)
        
//        printf("Theiler type %d, for scale %d and legacy Theiler %d\n", sp->type, stride, sp->Theiler);
        sp->N_eff_max   = get_N_eff_from_mask(mask, npts, p, stride, lag, sp->Theiler, sp->N_real);
        if (sp->N_eff<1) sp->N_eff = sp->N_eff_max/2; // 2023-12-01, completely arbitrary (same as non-mask version)
    }
    else
    if (sp->type==4)                                // automatic Theiler style 4
    {   sp->Theiler     = stride;
        sp->N_eff_max   = get_N_eff_from_mask(mask, npts, p, stride, lag, sp->Theiler, sp->N_real);
        
        if (do_auto_Neff==1)        sp->N_eff = sp->N_eff_max;  // N_eff "automatic" => we use the largest value
        if (sp->N_eff<2)            return(-1);
    
        if (sp->N_eff_max<sp->N_eff)                // we do not have enough points
        {                                           // we reduce tau_Theiler
            if (sp->N_eff_max>1)
            {   ratio = sp->N_eff/sp->N_eff_max;   
                if (ratio<1.3) ratio=1.3;           // first peculiar situation where sp->N_eff and sp->N_eff_max are similar
                sampling_ratio = (int)ceil(ratio);
                sp->Theiler = (int)(sp->Theiler/sampling_ratio/1.3);
                if (lib_verbosity>2)    
                    printf("[set_sampling_parameters_mask] tau Theiler reduction: (ratio %1.3f) now %d\n", ratio, sp->Theiler);
                
                if (sp->Theiler>1)
                {   sp->type=0;
                    sp->N_eff_max = set_sampling_parameters_mask(mask, npts, p, stride, lag, sp, func_name); // re-run with fixed Theiler
                    sp->type=4;
                }
                else 
                {   if (lib_verbosity>=0) 
                        return(print_error("set_sampling_parameters_mask","it seems even tau_Theiler=2 is not good enough..."));
                }
            }
            else                                    // added 2023-12-01
            {   if (lib_verbosity>=0) return(print_error("set_sampling_parameters_mask","no points available"));
            }
        }
        else                                        // there are enough points: we may increase the Theiler scale
        {
            ratio = sp->N_eff_max/sp->N_eff; 
            if (ratio>1.3) 
            {   sampling_ratio = (int)ceil(ratio);
                sp->Theiler = (int)(sp->Theiler*sampling_ratio/1.3);
                if (lib_verbosity>2)    
                    printf("[set_sampling_parameters_mask] tau Theiler increase: (ratio %1.3f) now %d\n", ratio, sp->Theiler);
                
                sp->type=0;
                sp->N_eff_max = set_sampling_parameters_mask(mask, npts, p, stride, lag, sp, func_name); // re-run with fixed Theiler
                sp->type=4;
                if (sp->N_eff_max<sp->N_eff)        // the increase was not successful => we "undo" it
                {   sp->Theiler = (int)(sp->Theiler/sampling_ratio);
                    if (lib_verbosity>2)    
                        printf("[set_sampling_parameters_mask] tau Theiler reduction: (ratio %1.3f) now %d\n", ratio, sp->Theiler);
                    sp->type=0;
                    sp->N_eff_max = set_sampling_parameters_mask(mask, npts, p, stride, lag, sp, func_name); // re-run with fixed Theiler
                    sp->type=4;
                }
            }
        }
        sp->Theiler_max = sp->Theiler;
    }
    
    return(sp->N_real_max);
} /* end of function "set_sampling_parameters_mask" */


