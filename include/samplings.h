/****************************************************************************************/
/* samplings.h                                                                          */
/*                                                                                      */
/* functions for embedding, and more generaly sampling, of datasets.                    */
/*                                                                                      */
/* Created by Nicolas Garnier on 2021/12/17.                                            */
/* Copyright 2010-2021 ENS-Lyon - CNRS. All rights reserved.                            */
/*                                                                                      */
/* 2021-12-17 : functions "Theiler_embed" and "increments"                              */
/* 2021-12-21 : function "increments_mask"                                              */
/****************************************************************************************/
#ifndef SAMPLINGS_H
#define SAMPLINGS_H

#define THEILER_2D_MINIMAL 1 // straightforward generalization of 1-d case: too minimal!
#define THEILER_2D_MAXIMAL 2 // simple adaptation by using tau = max(tau_x,tau_y): may not be enough
#define THEILER_2D_OPTIMAL 4 // ideal adaptation: tau = sqrt(tau_x^2 + tau_y^2): beware the rounding

#include <gsl/gsl_permutation.h>

struct sampling_parameters {
    int Theiler;        // Theiler scale
    int Theiler_max;    // max (if applicable)
    int N_eff;
    int N_eff_max;      // max (if applicable)
    int N_real;
    int N_real_max;     // max (used depending on type)
    int type;           // Theiler sampling type 
};
typedef struct sampling_parameters samp_param;

struct sampling_parameters_extra_2d {
    int last_Theiler_x; // Theiler scale in x in the last computation
    int last_Theiler_y; // Theiler scale in y in the last computation
    int type;           // 2-d Theiler type
};
typedef struct sampling_parameters_extra_2d samp_param_2d;

// for increments of any order:
int *get_binomial(int order);

// variables below are defined in "sampling.c":
extern samp_param samp_default;     // for default values (used in Python)
extern samp_param last_samp;        // values computed and used in the last function call
extern samp_param_2d samp_2d;       // values computed and used in the last function call (2d)

/********************************************************************************************************************/
// functions to perform sampling operations
// - 1-d:
void Theiler_embed(double *x, int npts, int mx, int px, int tau, int Theiler, size_t *ind, double *x_new, int npts_new);
void increments   (double *x, int npts, int mx, int px, int tau, int Theiler, size_t *ind, double *x_new, int npts_new);
void incr_avg     (double *x, int npts, int mx, int px, int tau, int Theiler, size_t *ind, double *x_new, int npts_new);
// - 2-d (2022):
void Theiler_embed_2d(double *x, int npts_x, int npts_y, int m, int p, int stride_x, int stride_y, int Theiler_x, int Theiler_y,
                        size_t *ind_x, size_t *ind_y, double *x_new, int npts_x_new, int npts_y_new);
void increments_2d   (double *x, int npts_x, int npts_y, int m, int p, int stride_x, int stride_y, int Theiler_x, int Theiler_y, 
						size_t *ind_x, size_t *ind_y, double *x_new, int npts_x_new, int npts_y_new);
void incr_avg_2d     (double *x, int npts_x, int npts_y, int m, int p, int stride_x, int stride_y, int Theiler_x, int Theiler_y, 
						size_t *ind_x, size_t *ind_y, double *x_new, int npts_x_new, int npts_y_new);
// - with masks (2022):
void Theiler_embed_mask(double *x, int npts, int mx, int px, int stride, size_t *ind_epoch, double *x_new, int npts_new);
void increments_mask   (double *x, int npts, int mx, int px, int stride, size_t *ind_epoch, double *x_new, int npts_new);
void incr_avg_mask     (double *x, int npts, int mx, int px, int stride, size_t *ind_epoch, double *x_new, int npts_new);


/********************************************************************************************************************/
// functions to handle enhanced sampling parameters
// - shuffling:
gsl_permutation *create_unity_perm(size_t N);
void shuffle_perm  (gsl_permutation *perm);
void free_perm     (gsl_permutation *perm);
// legacy embedding, used in the python module "tools" (in file "samplings_basic.c")
void time_embed(double *data, double *output, int nb_pts, int nb_pts_new, int nb_dim, int n_embed, int n_embed_max, int stride, int i_start, int n_window);
// - usefull formulas:
int Theiler_nb_pts_new         (int npts,        int stride, int n_embed_max);                  // in file "samplings_basic.c"
int compute_T_real                       (int p, int stride, int tau_Theiler, int N_eff);
int compute_N_real_max         (int npts, int p, int stride, int tau_Theiler, int N_eff);
// - dealing with sampling parameters:
void printf_sampling_parameters(samp_param sp, char *text);
int  check_sampling_parameters (int npts, int p, int stride, samp_param *sp, char *func_name);
int set_sampling_parameters    (int npts,       int p, int tau,                    samp_param *sp, char *func_name);
int set_sampling_parameters_2d (int nx, int ny, int p, int stride_x, int stride_y, samp_param *sp_x, samp_param *sp_y, char *func_name);

#endif
