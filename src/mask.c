/*
 *  mask.c
 *  
 *  Created by Nicolas Garnier on 2013/06/20.
 *  Copyright 2013 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2013-06-20: new source file containing all functions related with masking and epochs
 *  2020-02-23: new function "combine_masks"
 */

#include <math.h>       /* for fabs and nan operations */
#include <string.h>
#include <gsl/gsl_statistics_double.h>   // for mean and std

#include "samplings.h"              // for sampling functions
#include "verbosity.h"              // for verbosity
#include "library_matlab.h"         // compilation for Matlab
#include "math_tools.h"
#include "timings.h"

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
/********************************************************************************/
void combine_masks(char *mask_1, char *mask_2, int npts, char *mask_out)
{   register int i;
    
    for (i=0; i<npts; i++) mask_out[i] = mask_1[i] | mask_2[i]; // 2020-02-28: "|" or "+"
}



/********************************************************************************/
/* search for largest compact intervals in a mask :                             */
/*                                                                              */
/* mask      contains the mask                                                  */
/* npts      is the size of mask (nb of points in time)                         */
/* p         is the desired embedding dimension                                 */
/* stride    is the time lag for embedding                                      */
/* lag       is the time lag for TE (set it to 0 if using entropy or MI)        */
/*          T_min is computed inside now :                                      */
/*          T_min = p*stride for entropy, MI et al    (conservative!)           */
/*          T_min = p*stride+lag for TE               (conservative!)           */
/* i_window  indicates the number of the windo to consider (0<=i_win<stride)    */
/* ind_epoch (returned value) will contain the indices of beginning of epochs   */
/*                                                                              */
/* returned value (int) is the number of relevant epochs found                  */
/*                                                                              */
/*                                                                              */
/* 2013-06-17 : first, working version, returning max,min epochs sizes          */
/* 2013-06-18 : enhanced version, returning epochs                              */
/* 2016-05-17 : improved robustness                                             */
/* 2016-11-15 : some more checks. Note that T_min has to be choosen carefully!  */
/* 2016-11-16 : improved robustness, while simplifying                          */
/* 2019-02-03 : this function is too restrictive => forked into a new one       */
/* 2019-02-03 : forked from "analyse-mask"                                      */
/*              FYI, a set is point where embedding can be performed            */
/********************************************************************************/
int analyze_mask_conservative(char *mask, int npts, int p, int stride, int lag, int i_window, int **ind_epoch)
{   register int i=0, j=0, epoch=0, n_sets=0;
    int     *ind_epoch_tmp; // times of beginning of epochs
    int     *T_epoch_tmp;   // epochs durations
    char    is_OK = 1;
    int     T_min; // computed below, not outside the function anymore (2019-02-03)
    int     nb_pts_available, N_epoch = 0;
    
//    T_min = p*stride+lag;     // 2020-02-27: replaced by line below
//    T_min = (p-1)*stride+lag; // 2022-05-27: replaced by line below
    T_min = (p-1)*stride+1+lag;
    printf("[analyze_mask_conservative] : npts=%d, p=%d, stride=%d, i_win=%d, T_min=%d\n",
            npts, p, stride, i_window, T_min);
    
    ind_epoch_tmp = (int*)calloc((int)(npts), sizeof(int)); // 2016-05-17, allocate maximally, just in case
    T_epoch_tmp   = (int*)calloc((int)(npts), sizeof(int)); // 2016-05-17, allocate maximally, just in case

    i=0; // indice of time, we start scanning at the beginning
    j=0; // number of the epoch under study, or the next one
    
    while (is_OK)
    {   while ( (mask[i]==0) && (i<npts) )
        {   i++;
        }
        ind_epoch_tmp[j] = i;
    
        while ( (mask[i]>0)  && (i<npts) )
        {   i++;
        }
        
        if (i<npts)
        {   is_OK=1;
            T_epoch_tmp[j] = i - ind_epoch_tmp[j];
        }
        else
        {   is_OK=0;
            if (mask[i-1]>0)    T_epoch_tmp[j] = i - ind_epoch_tmp[j];
        }
        if (T_epoch_tmp[j]>=T_min)  // the epoch we found can be kept (it is long enough)
        {    j++;
#ifdef DEBUG
            printf("epoch %d : starts at time %d and last %d pts\n", j-1, ind_epoch_tmp[j-1], T_epoch_tmp[j-1]);
#endif
        }
    }
    
    N_epoch = j;
#ifdef DEBUG
    printf("[analyze_mask_conservative] first stage: %d epochs found.\n", j);
#endif

    // 2019-02-03, now, we convert this set of epochs into the new format:

    // we count the number of satisfying sets:
    n_sets=0;
    for (epoch=0; epoch<N_epoch; epoch++)  // loop over epochs
    {   nb_pts_available = (T_epoch_tmp[epoch]-T_epoch_tmp[epoch]%stride)/stride - (p-1); // size of a single dataset in epoch i
        n_sets += nb_pts_available; 
        T_epoch_tmp[epoch] = nb_pts_available; // this erases the old value of T_epoch_tmp, and records the nb_pts_available instead
    }
 
    if (n_sets>0)   *ind_epoch = (int*)calloc(n_sets,sizeof(int)); // allocate exactly
    else            return(-1);
    
// second copy:
    j=0;
    for (epoch=0; epoch<N_epoch; epoch++)  // loop over epochs
    {   for (i=0; i<T_epoch_tmp[epoch]; i++) // loop over points in 1 window and 1 epoch
            {   (*ind_epoch)[j] = i_window + ind_epoch_tmp[epoch] + stride*i;
                j++;
            }
    }

    if (j!=n_sets) printf("I've counted %d vs n_sets = %d !!!\n", j, n_sets);
            // useless test, but let's be careful

    free(ind_epoch_tmp);
    free(T_epoch_tmp);
    
    return(n_sets);
} /* end of function "analyze_mask_conservative" */





/********************************************************************************/
/* search mask for sets of indices to work on                                   */
/*                                                                              */
/* mask      contains the mask                                                  */
/* npts      is the size of mask (nb of points in time)                         */
/* p         is the desired embedding dimension                                 */
/* stride    is the time lag for embedding                                      */
/* lag       is the time lag for TE  (set it to 0 if useless)                   */
/* i_window  indicates which window (from 0 to stride-1) to consider            */
/*           (so it assumes a Theiler hypothesis)                               */
/* ind_epoch (returned value) will contain the indices of beginning of sets     */
/* returned value (int) is the number of relevant sets found                    */
/* 2020-02-27 : FYI, a set is a point where embedding can be performed          */
/*                                                                              */
/*                                                                              */
/* 2019-02-03 : forked from "analyse_mask", note: lag is not coded!             */
/* 2023-02-15 : parameter lag now explicitly discarded iin code below           */
/*              (to silent warnings)                                            */
/********************************************************************************/
int analyze_mask_Theiler_optimized(char *mask, int npts, int p, int stride, int lag, int i_window, int **ind_epoch)
{   register int i=0, j=0, n_sets=0;
    int     *ind_epoch_tmp; // times of beginning of epochs
    int     *T_epoch_tmp;   // epochs durations
    char    is_good_set;
    
    (void)lag;      // just to silent the compiler
    
    *ind_epoch = NULL;
    
    if ((i_window<0) || (i_window>=stride))
    {   printf("[analyze_mask_Theiler_optimized] bad window index %d\n", i_window);
        return(-1);
    }
    
    ind_epoch_tmp = (int*)calloc((int)(npts), sizeof(int)); // 2016-05-17, allocate maximally, just in case
    T_epoch_tmp   = (int*)calloc((int)(npts), sizeof(int)); // 2016-05-17, allocate maximally, just in case

    i=0;        // indice of time, we start scanning at the beginning
    n_sets = 0; // number of the set under study, or the next one
    
    for (i=i_window; i<(npts-(p-1)*stride); i+=stride) // scan all available begining points for embedding domains
                                                    // Theiler protocol is applied (sampling every stride)
    {
        is_good_set = 1;
        j = 0;
        while ( is_good_set && (j<p) )
        {   is_good_set = (mask[i+j*stride]>0) ? 1 : 0;
            j++;
        }
        
        if (is_good_set>0)
        {   ind_epoch_tmp[n_sets] = i;
            n_sets++;
#ifdef DEBUG
            printf("set %d : starts at time %d\n", n_sets-1, ind_epoch_tmp[n_sets-1]);
#endif
        }
    }
    // 2016-11-16, removed final tests for last point, as it is now included in the loop
#ifdef DEBUG
    printf("[analyze_mask_Theiler_optimized] %d sets found.\n", n_sets);
#endif
    if (n_sets>0)
    {   *ind_epoch = (int*)calloc(n_sets,sizeof(int)); // allocate exactly
        memcpy(*ind_epoch, ind_epoch_tmp, n_sets*sizeof(int));
    }
    
    free(ind_epoch_tmp);
    free(T_epoch_tmp);
    
    return(n_sets);
} /* end of function "analyze_mask_Theiler_optimized" */



/********************************************************************************/
/* search within a mask for sets of indices to work on                          */
/*                                                                              */
/* mask               contains the mask                                         */
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
        while ( is_good_set && (j<p) )          // check causal embedding
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
    
    free(ind_epoch_tmp);
    return(n_sets);
} /* end of function "analyze_mask_for_sampling" */



/********************************************************************************/
/* search a set of indices for optimal sampling                                 */
/*                                                                              */
/* ind_epoch   contains the indices of good epochs                              */
/* npts        is the size of ind_epoch                                         */
/* tau_Theiler contains the Theiler sampling time                               */
/*                                                                              */
/* N_eff_max   will be returned with values of N_eff (max) in each realization  */
/*              must be pre-allocated!                                          */
/* returned value (int) is the total number of effective points                 */
/*                                                                              */
/* 2022-05-28 : new function                                                    */
/* 2022-06-06 : now unused                                                      */
/********************************************************************************/
int analyze_epochs_for_sampling(int *ind_epoch, int npts, int tau_Theiler, int N_real, int *N_eff_max)
{   register int i_start, i, j, N_eff_tot=0;
//    *N_eff_max=(int*)calloc(N_real, sizeof(int));
    
    for (i_start=0; i_start<N_real; i_start++) // loop on realizations
    {   i=i_start;                  // index in the original dataset
        j=0;                        // index on the epochs
        while (i<=ind_epoch[npts-1]) // an efficient estimate of the size of the original dataset
        {   while (ind_epoch[j] < i) j++;
            if (ind_epoch[j] == i) N_eff_max[i_start]++;
            i+=tau_Theiler;
        }
        //then we sum, just for our own stats:
        N_eff_tot += N_eff_max[i_start];
    }
    
    printf("[analyze_epochs_for_sampling] : contents of N_eff_max\n");
    for (i=0; i<N_real; i++) printf("%d ", N_eff_max[i]);
    printf("\n");

    return(N_eff_tot);
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
/*                  ==-1 or -2 are not supported for masks                              */
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
/****************************************************************************************/
int set_sampling_parameters_mask(char *mask, int npts, int p, int stride, int lag, samp_param *sp, char *func_name)
{   register int j;
#define RANDOMNESS 3 
    int do_auto_Neff=0;
	int *N_eff_max=NULL, sampling_ratio=1;
	int *i_real_good=NULL; // 2022-10-24 : to record where are the "good" starting points of realizations with large enough N_eff_max
	double ratio;
//double temps;

    if (sp->Theiler>0)          sp->type=0;         // tau_Theiler is imposed (this is indeed a viable option)
    else if (sp->Theiler==-4)   sp->type=4;         // 4: we may reduce Theiler to have the required N_eff
    else                        sp->type=3;         // automatic based on uniform sampling + Theiler adaptation
    if (sp->type>0)             sp->Theiler=stride;
    if (sp->N_eff<1)            do_auto_Neff=1;     // for (legacy)
    if (sp->N_real==-1)         sp->N_real=stride;  // for legacy automatic
    else if (sp->N_real<1)      sp->N_real=1;       // new default value for automatic
            
    sp->Theiler_max = sp->Theiler;                  // the max has (for now) no meaning with masks
    sp->N_real_max = sp->N_real;                    // no shuffling of realizations with masks
    
//temps=0; tic_in();        
    N_eff_max   = (int*)calloc(sp->N_real, sizeof(int));
    i_real_good = (int*)calloc(sp->N_real, sizeof(int));
    // 2022-10-26: remark: the loop below should be with "stride" steps, and not "sp->N_eff"
    for (j=0; j<sp->N_real; j++)
    {   N_eff_max[j] = analyze_mask_for_sampling(mask+j, npts-j, p, stride, lag, sp->Theiler, 0, NULL);
//        printf("   %d ", N_eff_max[j]);
    }
//    printf("for scale %d and Theiler %d\n", stride, sp->Theiler);
//toc_in(&temps);
//printf("%d real => time %f\t(%f per real)\n", sp->N_real, temps, temps/sp->N_real);
    
    sp->N_eff_max = find_min_int(N_eff_max, sp->N_real);
    free(N_eff_max);
//    printf("\t => sp->N_eff_max = %d\n", sp->N_eff_max);
    
    if (do_auto_Neff==1)        sp->N_eff = sp->N_eff_max;  // N_eff "automatic" => we use the largest value
    if (sp->N_eff<2)            return(-1);
    
    if (sp->N_eff_max<sp->N_eff)
    {   if ( (sp->type==4) && (sp->N_eff_max>1) )       // we reduce tau_Theiler to have a large enough N_eff 
        {   ratio = sp->N_eff/sp->N_eff_max;   
            if (ratio<1.3) ratio=1.3;                   // first peculiar situation where sp->N_eff and sp->N_eff_max are similar
            sampling_ratio = (int)ceil(ratio);
            sp->Theiler = (int)(sp->Theiler/sampling_ratio/1.3);
            if (sp->Theiler<1)
            {   printf("[set_sampling_parameters_mask] error! cannot adapt Theiler scale\n");
                printf("\t  N_eff_min = %d but N_eff = %d is too large\n", sp->N_eff_max, sp->N_eff);
                if (lib_verbosity>1)
                    printf("\t-> try to increase the number of points in the dataset, or reduce N_eff, or set it automatic.\n");
                return(-1);
            }
            else 
            {   return(set_sampling_parameters_mask(mask, npts, p, stride, lag, sp, func_name));
            }
        }
        else
        {   printf("[set_sampling_parameters_mask] error! N_eff_min = %d but N_eff = %d is too large\n", sp->N_eff_max, sp->N_eff);
            if (lib_verbosity>1)
                printf("\t-> try to increase the number of points in the dataset, or reduce N_eff, or set it automatic.");
            return(-1);
        }
    }
    else if ((sp->N_eff_max/sp->N_eff) > RANDOMNESS)    // we have too many points
    {   ratio = (sp->N_eff_max/sp->N_eff);              // at least = RANDOMNESS
        sp->Theiler   *= (int)trunc(ratio/1.3);
//        printf("\ratio = %f new sp->Theiler = %d\n", ratio, sp->Theiler );
        sp->N_eff_max /= (int)ceil(ratio*1.3);                        // we under-estimate N_eff_max (security)
        // 2022-06-07: note that a better option would be to re-run "set_sampling_parameters_mask" 
        // with imposed Theiler = the new Theiler value, but this may be timme consuming if npts is large
        // the present solution is under carefull study...
    }

    free(i_real_good);
    return(sp->N_real_max);
} /* end of function "set_sampling_parameters_mask" */


