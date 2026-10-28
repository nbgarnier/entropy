/*
 *  surrogates_ann_PMI.h
 *  
 *
 *  Created by Nicolas Garnier on 2026/10/27.
 *  Copyright 2012-2026 ENS-Lyon - CNRS. All rights reserved.
 *
 *
 *  2026-10-27 : file created
 */

/* low-level function to generate surrogates for partial mutual information	*/
int surrogates_local_perm(double *x, int nx, int m, int p, int q, int k_perm, int N_surrogates, double *surrogates);

/* low level utility function to test if i is in set                        */
int is_in_set(int i, int *set, int npts);

