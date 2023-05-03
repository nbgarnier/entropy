/*
 *  mask.h
 *  
 *
 *  Created by Nicolas Garnier on 2013/06/20.
 *  Copyright 2013 ENS-Lyon - CNRS. All rights reserved.
 *
 *  Masking facilities (epochs)
 *
 */
#include "samplings.h"

// to fill a mask corresponding to NaN or finite values:
int NaN_mask                        (double *x, int npts, int m, char *mask);
int finite_mask                     (double *x, int npts, int m, char *mask);

// to combine two masks into a single one:
void combine_masks                  (char *mask_1, char *mask_2, int npts, char *mask_out);

// new function, equivalent (again specialized in "epochs"):
int analyze_mask_conservative       (char *mask, int npts, int p, int stride, int lag, int i_window, int **ind_epoch);
// new function, that aggregate all possible points, in each window (so specialized in NaN treatment):
int analyze_mask_Theiler_optimized  (char *mask, int npts, int p, int stride, int lag, int i_win, int **ind_epoch);

// for (random) samplings:
int analyze_mask_for_sampling       (char *mask, int npts, int p, int stride, int lag, int Theiler, int do_return_indices, int **ind_epoch);
int analyze_epochs_for_sampling     (int *ind_epoch, int npts, int tau_Theiler, int N_real, int *N_eff_max);
int set_sampling_parameters_mask    (char *mask, int npts, int p, int stride, int lag, samp_param *sp, char *func_name);
