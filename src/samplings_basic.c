/****************************************************************************************/
/* sampling_basic.c                                                                     */
/*                                                                                      */
/* functions for general manipulation, including sub-sampling, of datasets.             */
/*                                                                                      */
/* Created by Nicolas Garnier on 2021/12/17.                                            */
/* Copyright 2010-2024 ENS-Lyon - CNRS. All rights reserved.                            */
/*                                                                                      */
/* 2021-12-17 : function "Theiler_embed"                                                */
/* 2022-04-17 : function "set_sampling_parameters"                                      */
/* 2022-12-07 : fully rewritten function "set_sampling_parameters"                      */
/* 2024-09-25 : split from "samplings.c", keeping same header file                      */
/****************************************************************************************/
#include <math.h>                   // for trunc (to replace %)

#include "verbosity.h"
#include "samplings.h"



// old formula, dating back to old samplings; still usefull for regular embedding
int Theiler_nb_pts_new(int npts, int stride, int n_embed_max)
{   return ( (npts-npts%stride)/stride - (n_embed_max-1)); // size of a single dataset
}


/************************************************************************/
/* time-embedding  (legacy version)                                     */
/*                                                                      */
/* data       : initial data of size (nb_pts, nb_dim)                   */
/* output     : time-embedded datamust be allocated!                    */
/* nb_pts     : nb of points in time in the initial data                */
/* nb_pts_new : nb of points in time in the time-embedded data          */
/*                should be computed outside                            */
/* nb_dim     : nb of dimension of multivariate initial data            */
/* n_embed    : embedding dimension to use                              */
/* n_embed_max: max embedding dimension (for causal time reference)     */
/* stride     : time scale for embedding                                */
/* i_start    : (=i_window) select starting point:  0<=i_start<stride   */
/* n_window   : shift between realizations (if Theiler, then =stride)   */
/*                                                                      */
/* 2021-02-01 : new function, fun but useless...                        */
/* 2021-02-02 : tested OK, and much faster than pure python!            */
/************************************************************************/
void time_embed(double *data, double *output, int npts, int npts_new, int nb_dim, int n_embed, int n_embed_max, int stride, int i_start, int n_window)
{   register int i, l, d;

	for (i=0; i<npts_new; i++)    // loop over points in the i_start=i_window window
	for (l=0; l<nb_dim; l++)        // loop over existing dimensions in x
    for (d=0; d<n_embed; d++)       // loop over embedding 
        output[i + (d + l*n_embed)*npts_new] = data[l*npts + i_start + n_window*i + stride*(n_embed_max-1-d)];
	
    return;
}



/****************************************************************************************/
/* "crop" a pointer/array                                                               */
/* such that some points are dropped, either at the beginning or at the end             */
/*                                                                                      */
/* in, out  : input and output array, must both be allocated                            */
/* npts, n  : initial dimensions of input                                               */
/* npts_new : output dimension in the time direction                                    */
/* i_window : starting point in the time direction                                      */
/*                                                                                      */
/* 2024-09-25 : to allow AMI and entrpy rate estimation directly from Python            */
/****************************************************************************************/
void crop_array(double *data, double *output, int npts, int n, int npts_new, int i_window)
{   register int i,l;

    if ((i_window + npts_new) > npts) 
    {   printf("[crop_array] error, bad parameters\n");
        return;
    }

    for (i=0; i<npts_new; i++)
    for (l=0; l<n; l++)
    {   output[i + l*npts_new] = data[i_window+i + l*npts];
    }

    return;
}
