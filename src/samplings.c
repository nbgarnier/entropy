/****************************************************************************************/
/* sampling.c                                                                           */
/*                                                                                      */
/* functions for embedding, and more generaly sampling, datasets.                       */
/*                                                                                      */
/* Created by Nicolas Garnier on 2021/12/17.                                            */
/* Copyright 2010-2022 ENS-Lyon - CNRS. All rights reserved.                            */
/*                                                                                      */
/* 2021-12-17 : function "Theiler_embed"                                                */
/* 2022-04-17 : function "set_sampling_parameters"                                      */
/* 2022-12-07 : fully rewritten function "set_sampling_parameters"                      */
/****************************************************************************************/
#include <stddef.h>                 // for size_t
#include <stdio.h>                  // for printf
#include <math.h>                   // for trunc (to replace by %)

#include <gsl/gsl_rng.h>            // for random permutations
#include <gsl/gsl_randist.h>

#include "verbosity.h"
#include "samplings.h"

int binomial_1[2] = {1, -1};
int binomial_2[3] = {1, -2,  1};
int binomial_3[4] = {1, -3,  3,  -1};
int binomial_4[5] = {1, -4,  6,  -4,  1};
int binomial_5[6] = {1, -5, 10, -10,  5, -1};
int binomial_6[7] = {1, -6, 15, -20, 15, -6, 1};  

int *get_binomial(int order)
{   if      (order==2) return binomial_1; // a trick
    else if (order==3) return binomial_2;
    else if (order==4) return binomial_3;
    else if (order==5) return binomial_4;
    else if (order==6) return binomial_5;
    else if (order==7) return binomial_6;
    else return NULL;
}

// global level defaults:
samp_param samp_default = { .Theiler=-4, .N_eff=4096, .N_real=10, .type=4}; 
// last used values:
samp_param last_samp = { .Theiler=0, .Theiler_max=0, .N_eff=0, .N_eff_max=0, .N_real=0, .N_real_max=0, .type=0}; 
samp_param_2d samp_2d = {.type=THEILER_2D_MAXIMAL};

static int realizations_rng_state=0; // to keep track of the rng state
static gsl_rng *realizations_rng;
    
/****************************************************************************************/
/* to perform embedding of data according to Theiler prescription                       */
/*                                                                                      */
/* used by "compute_entropy_ann"                                                        */
/*         "compute_relative_entropy_ann"                                               */
/*         "compute_mutual_information_ann"                                             */
/*          and so on...                                                                */
/*                                                                                      */
/* 2021-12-17 : new function                                                            */
/****************************************************************************************/
void Theiler_embed_old(double *x, int npts, int mx, int px, int tau, int Theiler, double *x_new, int npts_new)
{   register int i,                 // indices for time positions
                 d, l;              // indices for dimensionality
    
    for (i=0; i<npts_new; i++)      // loop on time in 1 window
        {   for (d=0; d<mx; d++)    // loop on existing dimensions in x
            for (l=0; l<px; l++)    // loop on embedding
                 x_new[i + npts_new*( d + l*mx )] = x[Theiler*i - l*tau + d*npts];
        }

    return;
}

void Theiler_embed(double *x, int npts, int mx, int px, int tau, int Theiler, size_t *ind, double *x_new, int npts_new)
{   register int i,                 // indices for time positions
                 d, l;              // indices for dimensionality
    
    for (i=0; i<npts_new; i++)      // loop on time in 1 window
        {   for (d=0; d<mx; d++)    // loop on existing dimensions in x
            for (l=0; l<px; l++)    // loop on embedding
                 x_new[i + npts_new*( d + l*mx )] = x[Theiler*ind[i] - l*tau + d*npts];
        }

    return;
}

void Theiler_embed_mask(double *x, int npts, int mx, int px, int tau, size_t *ind_epoch, double *x_new, int npts_new)
{   register int i,                 // indices for time positions
                 d, l,              // indices for dimensionality
                 s;                 // center
    
    for (i=0; i<npts_new; i++)      // loop on time in 1 window
    {   s = ind_epoch[i];
        for (d=0; d<mx; d++)        // loop on existing dimensions in x
        {   for (l=0; l<px; l++)    // loop on embedding
                 x_new[i + npts_new*( d + l*mx )] = x[s - l*tau];
            s += npts;
        }
    }

    return;
}


/****************************************************************************************/
/* to perform embedding of data according to Theiler prescription                       */
/* 2-d version of the function above                                                    */
/*                                                                                      */
/* 2022-01-07 : new function                                                            */
/* 2022-05-19 : made embedding "causal" (according to increasing indexes)               */
/****************************************************************************************/
void Theiler_embed_2d(double *x, int npts_x, int npts_y, int m, int p, int stride_x, int stride_y, int Theiler_x, int Theiler_y, size_t *ind_x, size_t *ind_y, double *x_new, int npts_x_new, int npts_y_new)
{   register int ix, iy, d, l;       
    register int npts     = npts_x*npts_y;
    register int npts_new = npts_x_new*npts_y_new;

    for (ix=0; ix<npts_x_new; ix++)     // loop over new points in x
    for (iy=0; iy<npts_y_new; iy++)     // loop over new points in y
        {   for (d=0; d<m; d++)         // loop on existing dimensions
            for (l=0; l<p; l++)         // loop on embedding
                (x_new + npts_new*( d + l*m )) [iy + npts_y_new*ix] 
                    = x[ (Theiler_x*ind_x[ix]*npts_y + Theiler_y*ind_y[iy])  -  l*(stride_x*npts_y + stride_y)  +  d*npts ];
        }

    return;
}



/****************************************************************************************/
/* to compute causal increments of data according to Theiler prescription               */
/*                                                                                      */
/* used by "compute_entropy_increments".                                                */
/*         "compute_regularity_index_ann"                                               */
/*          and so on...                                                                */
/*                                                                                      */
/* 2021-12-17 : new function                                                            */
/****************************************************************************************/
void increments_old(double *x, int npts, int mx, int px, int tau, int Theiler, double *x_new, int npts_new)
{   register int i,                 // indices for time positions
                 d, l;              // indices for dimensionality
    register int *binome=get_binomial(px);
       
    if (binome==NULL) return;    
    for (i=0; i<npts_new; i++)      // loop on time in 1 window
    {   for (d=0; d<mx; d++)        // loop on existing dimensions in x
        {   x_new[i + npts_new*d] = x[Theiler*i + d*npts];
            for (l=1; l<px; l++)    // loop on embedding (here, increment order)
                x_new[i + npts_new*d] += binome[l]*x[Theiler*i - l*tau + d*npts];
        }
    }

    return;           
}



/****************************************************************************************/
/* to compute causal increments of data according to Theiler prescription               */
/* same as "increments", but with indices given by ind (from a random ppermutation)     */
/*                                                                                      */
/* used by "compute_entropy_increments".                                                */
/*         "compute_regularity_index_ann"                                               */
/*          and so on...                                                                */
/*                                                                                      */
/* 2022-05-10 : new function                                                            */
/****************************************************************************************/
void increments(double *x, int npts, int mx, int px, int tau, int Theiler, size_t *ind, double *x_new, int npts_new)
{   register int i,                 // indices for time positions
                 d, l,              // indices for dimensionality
                 s;                 // center
    register int *binome=get_binomial(px);
       
    if (binome==NULL) return;
          
    for (i=0; i<npts_new; i++)      // loop on time in 1 window
    {   s = Theiler*ind[i];
        for (d=0; d<mx; d++)        // loop on existing dimensions in x
        {//   s = Theiler*ind[i] + d*npts; // 2022-05-17: optimzed for speed in 2 other lines
            x_new[i + npts_new*d] = x[s];
            for (l=1; l<px; l++)    // loop on embedding (here, increment order)
                x_new[i + npts_new*d] += binome[l]*x[s - l*tau];
            s+=npts;                // optimization
        }
    }

    return;           
}



/****************************************************************************************/
/* to compute averaged causal increments of data according to Theiler prescription      */
/*                                                                                      */
/*                                                                                      */
/* 2022-01-08 : new function, assuming p=2                                              */
/****************************************************************************************/
void incr_avg(double *x, int npts, int mx, int px, int tau, int Theiler, size_t *ind, double *x_new, int npts_new)
{   register int i, d, l, s, t;
    register int *binome=get_binomial(px);
       
    if (binome==NULL) return;
    
    for (i=0; i<npts_new; i++)      // loop on time in 1 window
    {   for (d=0; d<mx; d++)        // loop on existing dimensions in x
        {   s = Theiler*ind[i] + d*npts;
            x_new[i + npts_new*d] = x[s];
            for (l=1; l<px; l++)    // loop on embedding (here, increment order=2 fixed)
            for (t=1; t<=tau; t++)
                x_new[i + npts_new*d] += binome[l]*x[s - l*t]/tau;
        }
    }

    return;           
}



/****************************************************************************************/
/* to compute increments of data according to Theiler prescription                      */
/* 2-d version of the function above                                                    */
/*                                                                                      */
/* 2022-01-08 : new function                                                            */
/* 2022-05-19 : made increments "causal" (according to increasing indexes)              */
/****************************************************************************************/
void increments_2d(double *x, int npts_x, int npts_y, int m, int p, int stride_x, int stride_y, int Theiler_x, int Theiler_y, size_t *ind_x, size_t *ind_y, double *x_new, int npts_x_new, int npts_y_new)
{   register int ix, iy, d, l, s;       
    register int npts     = npts_x*npts_y;
    register int npts_new = npts_x_new*npts_y_new;
    register int *binome=get_binomial(p);
       
/*    printf("[increments_2d] called with stride = %d,%d and Theiler = %d,%d\n", stride_x,stride_y,Theiler_x,Theiler_y);
    printf("m=%d, p=%d\n", m,p);
      fflush(stdout);
*/      
    if (binome==NULL) return;
    for (ix=0; ix<npts_x_new; ix++)     // loop over new points in x
    for (iy=0; iy<npts_y_new; iy++)     // loop over new points in y
    {   s = Theiler_x*ind_x[ix]*npts_y + Theiler_y*ind_y[iy];
        for (d=0; d<m; d++)             // loop on existing dimensions
        {   (x_new + npts_new*d) [iy + npts_y_new*ix] = x[s + d*npts];
            for (l=1; l<p; l++)         // loop on embedding (here, increment order)
                (x_new + npts_new*d) [iy + npts_y_new*ix] 
                    += binome[l]*x[ s - l*(stride_x*npts_y + stride_y)  +  d*npts ];
        }
/*        if ( (x_new) [iy + npts_y_new*ix] != stride_x) 
        {   printf("\tpb with %d,%d -> inc = %f vs stride_x = %d\n", ix,iy, (x_new) [iy + npts_y_new*ix], stride_x); 
            printf("\tcenter pt : %f from index %d\n", x[s], s);
            printf("\tother pt  : %f from index %d\n", x[s + 1*(stride_x*npts_y + stride_y)], s + 1*(stride_x*npts_y + stride_y));
        } // debug test
*/ 
    }

    return;
}


/****************************************************************************************/
/* to compute increments of data according to Theiler prescription                      */
/* 2-d version of the function above                                                    */
/*                                                                                      */
/* 2022-01-12 : new function (untested)                                                 */
/* 2022-05-19 : made increments "causal" (according to increasing indexes)              */
/****************************************************************************************/
void incr_avg_2d(double *x, int npts_x, int npts_y, int m, int p, int stride_x, int stride_y, int Theiler_x, int Theiler_y, size_t *ind_x, size_t *ind_y, double *x_new, int npts_x_new, int npts_y_new)
{   register int ix, iy, d, l, s, tx, ty, t_norm;       
    register int npts     = npts_x*npts_y;
    register int npts_new = npts_x_new*npts_y_new;
    register int *binome=get_binomial(p);
       
    if (binome==NULL) return;
    t_norm = ((stride_x>0)? stride_x:1) * ((stride_y>0)? stride_y:1);

    for (ix=0; ix<npts_x_new; ix++)     // loop over new points in x
    for (iy=0; iy<npts_y_new; iy++)     // loop over new points in y
    {   s = Theiler_x*ind_x[ix]*npts_y + Theiler_y*ind_y[iy];
        for (d=0; d<m; d++)             // loop on existing dimensions
        {   (x_new + npts_new*d) [iy + npts_y_new*ix] = x[s + d*npts];
            for (l=1; l<p; l++)         // loop on embedding (here, increment order)
            for (tx=1; tx<=stride_x; tx++)
            for (ty=1; ty<=stride_y; ty++)
                (x_new + npts_new*d) [iy + npts_y_new*ix] 
                    += binome[l]*x[ s  -  l*(tx*npts_y + ty)  +  d*npts ]/t_norm;
        }
    }

    return;
}



/****************************************************************************************/
/* to compute increments of masked data according to Theiler prescription               */
/*                                                                                      */
/* used by "compute_entropy_increments_ann_mask"                                        */
/*         "compute_regularity_index_ann_mask"                                          */
/*          and so on...                                                                */
/*                                                                                      */
/* 2021-12-21 : new function                                                            */
/* 2022-06-06 : slightly rewritten to mimick function "increments"                      */
/****************************************************************************************/
void increments_mask(double *x, int npts, int mx, int px, int stride, size_t *ind_epoch, double *x_new, int npts_new)
{   register int i,                 // indices for time positions
                 d, l,              // indices for dimensionality
                 s;                 // center
    register int *binome=get_binomial(px);
       
    if (binome==NULL) return;    
    for (i=0; i<npts_new; i++)      // loop on time over acceptable points for the current window
    {   s = ind_epoch[i];
        for (d=0; d<mx; d++)        // loop on existing dimensions in x
        {   x_new[i + npts_new*d] = x[s];
            for (l=1; l<px; l++)    // loop on embedding (here, increment order)
                x_new[i + npts_new*d] += binome[l]*x[s - l*stride];
            s += npts;              // for next dimension
        }
    }
 
    return;           
}


void incr_avg_mask(double *x, int npts, int mx, int px, int stride, size_t *ind_epoch, double *x_new, int npts_new)
{   register int i,                 // indices for time positions
                 d, l,              // indices for dimensionality
                 s, t;              // center, and indice for stride
    register int *binome=get_binomial(px);
       
    if (binome==NULL) return;    
    for (i=0; i<npts_new; i++)      // loop on time over acceptable points for the current window
    {   s = ind_epoch[i];
        for (d=0; d<mx; d++)        // loop on existing dimensions in x
        {   x_new[i + npts_new*d] = x[s];
            for (l=1; l<px; l++)    // loop on embedding (here, increment order)
            for (t=1; t<=stride; t++)
                x_new[i + npts_new*d] += binome[l]*x[s - l*stride]/stride;
            s += npts;              // for next dimension
        }
    }
 
    return;           
}


/****************************************************************************************/
/* creates and returns a permutation of N elements                                      */
/*                                                                                      */
/* note that the permutation is allocated on the fly by this function,                  */
/* so it should be properly de-allocated later after use                                */
/****************************************************************************************/
gsl_permutation *create_unity_perm(size_t N)
{   gsl_permutation *perm;
    const gsl_rng_type *T;
    
    if (realizations_rng_state==0) // never used
    {   gsl_rng_env_setup();
        T = gsl_rng_default;
        realizations_rng = gsl_rng_alloc(T);
        realizations_rng_state++;
    }
    
    perm = gsl_permutation_alloc(N);
    gsl_permutation_init(perm);

    return(perm);
}

/****************************************************************************************/
/* creates a new permutation of N elements, in-place                                    */
/*                                                                                      */
/* note that the permutation should be already allocated                                */
/****************************************************************************************/
void shuffle_perm(gsl_permutation *perm)
{   if (realizations_rng_state>0)
        gsl_ran_shuffle(realizations_rng, perm->data, perm->size, sizeof(size_t));

    return;
}

/****************************************************************************************/
/* deallocates a permutation                                                            */
/*                                                                                      */
/* note that the permutation should be already allocated                                */
/****************************************************************************************/
void free_perm(gsl_permutation *perm)
{   if (realizations_rng_state>0) realizations_rng_state--;
/*    if (realizations_rng_state>0)
    {   gsl_rng_free(realizations_rng);
        realizations_rng_state = 0;
    }
*/  // 2022-05-24: I now postpone the cleaning of the rng to the code exiting
    gsl_permutation_free(perm);
    return;
}


void printf_sampling_parameters(samp_param sp, char *text)
{   printf("variable ""%s"":", text);
    printf("\tTheiler prescription : %d ", sp.type);
    switch (sp.type)
    {   case 0 :   printf("(no Theiler prescription)\n");
                    break;
        case 1 :   printf("(legacy: fixed Theiler + uniform sampling)\n");
                    break;
        case 2 :   printf("(adjusted Theiler + uniform sampling)\n");
                    break;
        case 3 :   printf("(fixed Theiler + random)\n");
                    break;
        case 4 :   printf("(adjusted Theiler + random)\n");
                    break;
        default :   printf("(unknown!)\n");
                    break;
    }
    printf("\t\tTheiler : %d\t(max : %d)\n", sp.Theiler,  sp.Theiler_max);
	printf("\t\tN_eff   : %d\t(max : %d)\n", sp.N_eff,    sp.N_eff_max); 
    printf("\t\tN_real  : %d\t(max : %d)\n", sp.N_real,   sp.N_real_max);
    printf("\n");
    return;
}


// XIII.209 & XIII.224, for fixed N_eff
int compute_T_real(int p, int stride, int tau_Theiler, int N_eff)
{   int T_embed = (p-1)*stride+1;
    return( (N_eff-1)*tau_Theiler + T_embed );
}
// XIII.212 & XIII.224, to be used with N_eff_max
int compute_N_real_max(int npts, int p, int stride, int tau_Theiler, int N_eff)
{   
    return( npts - compute_T_real(p, stride, tau_Theiler, N_eff) + 1 );  
}


/****************************************************************************************/
/* to check a set of sampling parameters for a given dataset and given embedding params */
/*                                                                                      */
/* used by the function  "check_sampling_parameters" below                              */
/* (where the parameter 'warning_level' is adjusted depending on the algorithm)         */
/*                                                                                      */
/* lib_warning_level:   0 : warnings are treated as errors                                  */
/*                  1 : warnings are not errors, but are printed                        */
/*                  2 : physical checks may raise additional warnings                   */
/****************************************************************************************/
int check_sampling_parameters(int npts, int p, int stride, samp_param *sp, char *func_name)
{   int T_real    = compute_T_real(p, stride, sp->Theiler, sp->N_eff_max);  // XIII.212
    int npts_left = npts - (sp->N_real_max - 1) - (p-1)*stride - 1;         // XIII.213
    char message[128];
    
    if (npts < (T_real + sp->N_real_max - 1))
    {   if (lib_verbosity>0) 
        {   sprintf(message, " [check_sampling_parameters] not enough points");
            print_error(func_name, message);
        }
        return(-1);
    }
    if (sp->N_eff > sp->N_eff_max)
    {   if (lib_verbosity>0) 
        {   sprintf(message, " [check_sampling_parameters] N_eff (%d) is larger than N_eff_max (%d)", sp->N_eff, sp->N_eff_max);
            print_error(func_name, message);
        }
        if (lib_verbosity>1) 
        {   if (sp->type==0) printf("\t -> you may try to reduce either your imposed N_eff or your imposed Theiler scale");
        }
        return(-2);        
    }
    if (sp->N_real > sp->N_real_max)
    {   if (lib_verbosity>0)
        {   sprintf(message, " [check_sampling_parameters] N_real = %d > N_real_max = %d", sp->N_real, sp->N_real_max);
            print_error(func_name, message);
        }
        return(-3);
    }
    if (sp->N_eff<1)
    {   if (lib_verbosity>0)
        {   sprintf(message, " [check_sampling_parameters] N_eff = %d < 1", sp->N_eff);
            print_error(func_name, message);
        }
        return(-4);
    }
    if (sp->N_real<1)
    {   if (lib_verbosity>0)
        {   sprintf(message, " [check_sampling_parameters] N_real = %d", sp->N_real);
            print_error(func_name, message);
        }
        return(-5);
    }
    if (npts_left<1)
    {   if (lib_verbosity>0) print_error(func_name, " [check_sampling_parameters] not enough points");
        return(-6);
    }
    
    if (sp->Theiler < stride) 
    {   if (lib_warning_level==2)
        {   if (lib_verbosity>0)
            {   sprintf(message, " [check_sampling_parameters] adjusted Theiler scale (%d) would be smaller than the stride (%d).", 
                        sp->Theiler, stride);
                print_error(func_name, message);
                if (lib_verbosity>1)
                printf("\t-> this may be because there are not enough points in the dataset.\n"
                        "\t-> you can choose Theiler method 4 instead of 2, and examine results carefully.\n"
                        "\t-> or you can reduce N_eff.\n");
                printf("Select a lower warning_level (0 or 1) to bypass this error.\n");
            }
            return(-7);
        }
        else if (lib_warning_level==1)
        {   if ( (lib_verbosity>1) && (sp->type!=4) ) // 2023-02-15: added condition on sp.type (so no warning if tau_Theiler adapted by method 4)
            {   sprintf(message, " [check_sampling_parameters] Theiler scale (%d) is smaller than the stride (%d).", sp->Theiler, stride);
                print_warning(func_name, message);
                printf("Select a lower warning_level (0) or a lower verbosity (0 or 1) to silence this warning.\n");
            }
        }
    }
/*  // comented out 2023-02-15
    if (lib_warning_level>=1)   // physical checks, treated as a warning
    {   if (sp->N_real_max<stride)
        {   if (lib_verbosity>2) 
            {   printf("[%s] : " ANSI_COLOR_BLUE "warning " ANSI_COLOR_RESET 
                        "N_real_max (%d) is smaller than stride (%d): realizations may not be independant.", 
                        func_name, sp->N_real_max, stride);
                printf("Select a lower warning_level (0) or a lower verbosity (0, 1 or 2) to silence this warning.\n");
            }
        }
        
    }
*/
    return(0);
}

/****************************************************************************************/
/* computes optimal parameters, depending on user's requirements                        */
/*                                                                                      */
/* npts           : nb of points in dataset                                             */ 
/* p              : embedding dimension                                                 */ 
/* stride         : analysis scale                                                      */
/* sp             : contains the desired sampling parameters                            */
/*                  note that only .Theiler, .N_eff and .N_real are examined            */
/*                  whereas .type is modified accordingly                               */
/*     sp.Theiler : Theiler sampling,                                                   */
/*                  ==0  requests NO Theiler prescription                               */
/*                          (in that case, N_real is set to 1)                          */
/*                  ==-1 requests autosetting (=stride)  + uniform sampling (legacy)    */
/*                  ==-2 requests smarter auto-detection + uniform sampling             */
/*                          (in that case, Theiler can be larger than stride)           */
/*                  ==-3 requests autosetting (=stride)  + random sampling              */
/*                  ==-4 requests autosetting (possibly <stride!) + random sampling     */
/*                          (this case, imposed N_eff may lead to reducing Theiler)     */
/*                  ==any positive value will be used as an imposed tau_Theiler value   */
/*                                                                                      */
/*     .N_eff     : nb of effective points to use, value ==-1 requests auto-detection   */ 
/*                                                                                      */
/*     .N_real    : nb of realizations, value ==-1 requests legacy auto-detection       */
/*                                      value ==0  requests 1 (a single) realization    */
/* func_name      : name of the calling function                                        */
/*                                                                                      */
/* on success, returns N_real_max, the number of available possible realizations        */
/* on failure, returns -1                                                               */
/*                                                                                      */
/* 2022-04-20: to do: add check involving k                                             */
/* 2022-04-21: no, k is not a sampling parameter; the check should occur outside        */
/* 2022-05-13: important: for this to be efficient, best is to:                         */
/*              - impose N_eff (this fixes the bias of the estimator)                   */
/*              - impose N_real (so no autodetection!)                                  */
/*              - let Theiler automatic (ie, Theiler=tau)                               */
/* 2022-05-17: refactored tests -> new function                                         */
/* 2022-05-30: no more automatic N_real, as this was somehow stupid                     */
/*              if provided N_real<1, then N_real=1 is selected automaticaly            */
/*              (except if ==-1 : then legacy behavior)                                 */
/* 2022-12-07: full refactoring of the code                                             */
/* 2023-02-15: now recomputing sp->N_eff_max for sp.Theiler==3                          */
/* 2023-11-17: added 2 tests for Theiler type 4  (to be cleaned)                        */
/****************************************************************************************/
int set_sampling_parameters(int npts, int p, int stride, samp_param *sp, char *func_name)
{   int tmp=-13;                    
	int npts_left=npts;
  
    if (sp->Theiler>0)          sp->type=0;         // tau_Theiler is imposed (this is indeed a viable option)
    else if (sp->Theiler==-1)   sp->type=1;         // for tau_Theiler=tau and uniform sampling (legacy)
    else if (sp->Theiler==-2)   sp->type=2;         // for a smarter adjustment (spanning all the dataset)
    else if (sp->Theiler==-3)   sp->type=3;         // for tau_Theiler=tau and random sampling
    else if (sp->Theiler==-4)   sp->type=4;         // for a reduced tau_Theiler<tau if satisfying N_eff demands it
    else if (sp->Theiler<-4)    return(-1);
  
    // we first study the requirements on N_real:
    if (sp->N_real<1)           sp->N_real=1;       // N_real should usualy be specified! not good to have it automatic
                                                    // new default value if <0
    sp->N_real_max = sp->N_real;                    // N_real_max will be computed below, at the end of the function
                                                    // but may be needed before that recomputation
 
    if (sp->Theiler==0)                             // very old method introduced on 2011-11-14 : no Theiler-correction !
    {	sp->type        = 0;
        sp->Theiler     = 1;                        // to be able to use imposed N_eff and N_real
        sp->Theiler_max = 1;                        // 2022-12-14
        sp->N_eff_max   = npts - (p-1)*stride;      // legacy setting when no Theiler prescription
        if (sp->N_eff<1) sp->N_eff = sp->N_eff_max;
        sp->N_real_max  = 1;                        // legacy setting when no Theiler prescription
        sp->N_real      = 1;                        // 2022-12-07: may not be necessary
    }
    else
    if (sp->type==0)                                // Theiler scale is imposed 
    {   sp->Theiler_max = sp->Theiler;              // 2022-12-14
        if (sp->N_real<1) sp->N_real=stride;
        sp->N_real_max  = sp->N_real;
        npts_left       = npts - (p-1)*stride;      // 2023-11-29, for the line below
        sp->N_eff_max   = (npts_left-npts_left%(sp->Theiler))/(sp->Theiler);  
        if (sp->N_eff<1) sp->N_eff = sp->N_eff_max; // N_eff automatic
    }
    else
    if (sp->type==1)                                // automatic Theiler style 1 (legacy, 2012-05-03)
    {   sp->Theiler     = stride;
        sp->Theiler_max = sp->Theiler;              // 2022-12-14
        sp->N_real      = stride;                   // 2022-12-07: this crucial line was missing!
        sp->N_real_max  = sp->N_real;
        sp->N_eff       = (npts-npts%stride)/stride - (p-1);    // the old "nx_new"N_eff = N_eff_max
        sp->N_eff_max   = sp->N_eff;
    }
    else
    if (sp->type==2)                                // automatic Theiler style 2
    {   sp->N_eff_max   = sp->N_eff;                // N_eff should be specified! XIII.213 / XIII.219

        npts_left       = npts - (sp->N_real_max-1) - (p-1)*stride - 1; // XIII.213,215
        sp->Theiler_max = (int)trunc((double)npts_left/(sp->N_eff-1));  // XIII.213,219, we enlarge tau_Theiler
        sp->Theiler     = sp->Theiler_max;          // 2022-12-14

        // we need to recompute N_real_max:         // XIII.209, or brouillon 13/05/2022
        sp->N_real_max  = compute_N_real_max(npts, p, stride, sp->Theiler, sp->N_eff_max);
    }
    else
    if (sp->type==3)                                // automatic Theiler style 3
    {   sp->Theiler     = stride;
        sp->Theiler_max = sp->Theiler;              // 2022-12-14, tau_Theiler is fixed (not adapted)
        sp->N_eff_max   = (int)trunc((double)(npts-sp->N_real)/(sp->Theiler)) - (p-2); // XIII.221
        
        if (sp->N_eff<1) sp->N_eff = sp->N_eff_max/2; // 2022-12-07, completely arbitrary
        // we need to recompute N_real_max, so let's do it:
        sp->N_real_max  = compute_N_real_max(npts, p, stride, sp->Theiler, sp->N_eff_max);
    }
    else
    if (sp->type==4)                                // automatic Theiler style 4
    {   if (sp->N_eff>0)                            // N_eff is imposed
        {   sp->N_eff_max   = sp->N_eff;
            npts_left       = npts - (sp->N_real_max-1) - (p-1)*stride - 1; // XIII.222, same as XIII.213,215
            if (npts_left<=0)                       // test added 2023-11-17
            {   if (lib_warning_level>0) printf("[set_sampling_parameters] (type 4) npts_left = %d\n", npts_left);
                return(-1);
            }
            sp->Theiler_max = (int)trunc((double)npts_left/(sp->N_eff-1));  // XIII.222, same as XIII.213,219, we adapt tau_Theiler
            sp->Theiler     = sp->Theiler_max;      // 2022-12-14
            if (sp->Theiler<=0)                     // test added 2023-11-17
            {   if (lib_warning_level>0) printf("[set_sampling_parameters] (type 4) sp->Theiler = %d\n", sp->Theiler);
                return(-1);
            }

            // 2023-02-15: should we recompute N_eff_max? cf brouillon 2023-02-15
//            printf("[set_sampling_parameters] : I'm tempted to change N_eff_max from %d", sp->N_eff);
//            printf(" to %d\n", 1 + (int)trunc((npts-sp->N_real_max-(p-1)*stride)/sp->Theiler));
            sp->N_eff_max   = 1 + (int)trunc((npts-sp->N_real_max-(p-1)*stride)/sp->Theiler); // cf brouillon 2023-02-15
        }
        else                                        // N_eff automatic
        {   sp->Theiler     = stride;               // cf XIII.222 for discussion
            sp->Theiler_max = sp->Theiler;          // 2022-12-14
            sp->N_eff_max   = (int)trunc((double)(npts-sp->N_real)/(sp->Theiler)) - (p-2);    // XIII.222
            sp->N_eff       = sp->N_eff_max;
        }

        // we need to recompute N_real_max:         // XIII.209, or brouillon 13/05/2022
        sp->N_real_max  = compute_N_real_max(npts, p, stride, sp->Theiler, sp->N_eff_max);
    }

    // then some checks and warnings:
    tmp = check_sampling_parameters(npts, p, stride, sp, func_name);

    if (tmp<0) return(tmp);
    return(sp->N_real_max);
}


/****************************************************************************************/
/*                                                                                      */
/* note that asking automatic selection of parameters is done by setting values to -1   */
/* in the x-direction for the corresponding parameter                                   */
/*                                                                                      */
/* note that imposing N_eff and N_real is also done the x-direction                     */
/* and that the values given there are interpreted as the "total" N_eff (in 2-d)        */
/* and the "total" N_real (in 2-d). The splitting in directions x,y is done             */
/* automaticaly by the following function                                               */
/****************************************************************************************/
int set_sampling_parameters_2d(int nx, int ny, int p, int stride_x, int stride_y, 
                    samp_param *sp_x, samp_param *sp_y, char *func_name)
{   size_t N_eff_imposed=0, N_real_imposed;
    int  sx=stride_x, sy=stride_y, ret=0;          
	char message[64];
    
    if ( sp_x->N_eff>0)                                     // N_eff is imposed via N_eff_x, we distribute it homogeneously
    {   N_eff_imposed = sp_x->N_eff;                        // N_eff was for the overall 2-d image
        sp_x->N_eff = (int)(sqrt((N_eff_imposed)*nx/ny));   // brouillon 2022/05/04
        sp_y->N_eff = (int)(sqrt((N_eff_imposed)*ny/nx));   // 2022-05-19: removed 'trunc' to allow a little more
        // 2022-05-13:  if THEILER_2D_MINIMAL, note that stride_x and stride_y may also be used here,  
        // as they impact where the sampling (via N_eff_x and N_eff_y) has to be done to really be homogeneous
        // following tests are for very asymetric images:
        if (sp_x->N_eff<=1) { sp_x->N_eff=1;  sp_y->N_eff=N_eff_imposed; }
        if (sp_y->N_eff<=1) { sp_y->N_eff=1;  sp_x->N_eff=N_eff_imposed; }
    }
    else
    {   printf("[set_sampling_parameters_2d] sp_x->N_eff = %d\n", sp_x->N_eff);
        sp_y->N_eff = sp_x->N_eff;
    }
    if (sp_x->N_real>0)                                     // N_real is imposed,we distribute it homogeneously
    {   N_real_imposed = sp_x->N_real;                      // N_realizations was for the overall 2-d image
        sp_x->N_real = (int)(trunc)(sqrt((N_real_imposed)*nx/ny));   // brouillon 2022/05/04
        sp_y->N_real = (int)(trunc)(sqrt((N_real_imposed)*ny/nx));
        
        if (sp_x->N_real<=1) { sp_x->N_real=1;  sp_y->N_real=N_real_imposed; }
        if (sp_y->N_real<=1) { sp_y->N_real=1;  sp_x->N_real=N_real_imposed; }
    }
    else
    {   printf("[set_sampling_parameters_2d] sp_x->N_real = %d\n", sp_x->N_real);
        sp_y->N_real = sp_x->N_real;
    }

/*    printf("test in 2d function (start)\n");
    print_samp_param(*sp_x);
    print_samp_param(*sp_y);
*/
    // tmp variables:   
    if (sp_x->Theiler<0)                         // automatic Theiler
    {   sp_y->Theiler   =  sp_x->Theiler;
    
        // auto-adjustment of tau_Theiler, if required:
        if (samp_2d.type==THEILER_2D_MINIMAL)
        {   sx = (stride_x>0) ? stride_x : 1;    // OK, unless we later impose N_eff
            sy = (stride_y>0) ? stride_y : 1;    // OK, unless we later impose N_eff
        }
        else if (samp_2d.type==THEILER_2D_MAXIMAL)
        {   sx = (stride_x > stride_y) ? stride_x : stride_y;
            sy = sx;
        }
        else if (samp_2d.type==THEILER_2D_OPTIMAL)
        {   sx = (int)ceil(sqrt((double)stride_x*stride_x + (double)stride_y*stride_y));
            sy = sx;
        }
        else 
        {   printf("[%s] : unable to understand Theiler prescription!\n", func_name);
            return(-1);
        }
    }
    else
    {   if (sx<=0) sx=1;
        if (sy<=0) sy=1;
    }
//    printf("%d, %d\n", sx,sy); fflush(stdout);
  
    sprintf(message, "%s in x", func_name);
    ret=set_sampling_parameters(nx, p, sx, sp_x, message);
    if (ret<1) 
    {   print_error(message, "error in parameters; aborting !");
        return(-1);
    }  
    if (sp_x->N_real<1) printf("[%s] : only %d realizations !\n", message, sp_x->N_real);
    
    sprintf(message, "%s in y", func_name);
    ret=set_sampling_parameters(ny, p, sy, sp_y, message);
    if (ret<1) 
    {   printf("[set_sampling_parameters_2d] : error in parameters; aborting !\n");
        return(-1);  
    }
    if (sp_y->N_real<1) printf("[%s] : only %d realizations !\n", message, sp_y->N_real);    
        
    return(sp_x->N_real_max*sp_y->N_real_max); // max nb of realizations (also return in sp)
}

