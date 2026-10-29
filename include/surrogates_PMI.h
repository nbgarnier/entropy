/*
 *  surrogates_ann_PMI.h
 *  
 *
 *  Created by Nicolas Garnier on 2026/10/27.
 *  Copyright 2012-2026 ENS-Lyon - CNRS. All rights reserved.
 *
 *
 *  2026-10-27 : file created
 *  2026-10-28 : new function for direct use in python
 */

/* low level utility function to test if i is in set                        */
int is_in_set(int i, int *set, int npts);

/* low-level function to generate surrogates for partial mutual information	*/
int surrogates_local_perm_engine(double *x, int nx, int m, int q, int k_perm, int N_surrogates, double *surrogates);

/* exposed function to generate surrogates for partial mutual information	*/
int surrogates_local_perm(double *x, double *z, int nx, int m, int q, int k_perm, int N_surrogates, double *surrogates);

