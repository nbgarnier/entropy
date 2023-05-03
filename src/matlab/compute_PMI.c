/*
matlab wrapper for the function compute_partial_MI_ann defined in entropy_ann.c
or its masked version "_mask" defined in entropy_ann_mask.c

usage : see .m in /bin/matlab/

2013-02-27 : new version with implemented multi-dimensional input data
2013-06-18 : implemented masking for epochs
2013-06-20 : modified normalization when masking (only points in relevant epochs are used to compute normalization constants)
2021-12-21 : multi-thread and new parameters conventions
*/

#include <math.h>
#include <stdio.h>
#include <string.h> // for memcpy
#include "mex.h"

#include "entropy_ann_N.h"        // for non-masking version
#include "entropy_ann_mask.h"   // for masking version
#include "entropy_ann_threads.h"
#include "library_commons.h"    // for global variables

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

#include "matlab_commons.c"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray  *prhs[])
{   double  *x=NULL, *y=NULL, *z=NULL;
    char    *mask=NULL;
    int     npts=-1, dim[6]={1,1,1,1,1,1}, stride=1;
    int     Theiler=-4, N_eff=2048, N_real=10;
    int     k=k_default, algo=1, n_cores=-1;
//    int     ret=0; /* value returned by C function; == 0 if OK */
    double  *xout1, *xout2, *out_std1, *out_std2, *out_nbe, *out_eff, *out_nbw;
    int    do_use_mask=0;

    /* verify the number of input parameters */
    if (nrhs < 3)   mexErrMsgTxt("There are at least 3 input arguments required.\n"
                    "Please specify at least 3 vectors or matrices x, y and z.\n"
                    "See help for more information.\n");
    if (nrhs > 6)   mexErrMsgTxt("You specified %d input parameters, this functions requires 3, 4, 5, 6 or 7 parameters. "
                    "See help for more information.\n");

    // input the 3 datasets (3 first parameters) :
    read_data_parameters(prhs[0], &npts, &dim[0], &x);
    read_data_parameters(prhs[1], &npts, &dim[1], &y);
    read_data_parameters(prhs[2], &npts, &dim[2], &z);
//  printf("x has dimension %d and %d in time, whereas y and z have dimension %d and %d.\n", dim[0], npts, dim[1], dim[2]);
  
    /* parameter 4 contains embedding parameters : */
    if (nrhs>3) read_embedding_parameters(prhs[3], &dim[3+0], &dim[3+1], &dim[3+2], &stride, NULL, NULL);
    
    /* parameter 5 contains sampling parameters : */
    if (nrhs>4) read_sampling_parameters(prhs[4], &Theiler, NULL, &N_eff, &N_real);

    /* parameter 6 contains algorithms parameters : */
    if (nrhs>5) read_algorithms_parameters(prhs[5], &k, &algo, &n_cores, NULL);
    	
    /* parameter 7 : the mask (to select epochs) */
    if (nrhs==7) read_mask(prhs[6], &mask, &do_use_mask, npts);

    /* returned values : estimate of MI by algo 1 and by algo 2 : */
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    xout1 = mxGetPr(plhs[0]);

    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    xout2 = mxGetPr(plhs[1]);

    // now the math part :
    if (do_use_mask==1) compute_partial_MI_ann_mask(x,y,z, mask, npts, dim, stride, k, xout1, xout2);
    else                compute_partial_MI_ann_N   (x,y,z,       npts, dim, stride, Theiler, N_eff, N_real, k, xout1, xout2);

    plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
    out_std1 = mxGetPr(plhs[2]);  out_std1[0] = last_std;  // std of the estimation
    plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
    out_std2 = mxGetPr(plhs[3]);  out_std2[0] = last_std2;  // std of the estimation
    plhs[4] = mxCreateDoubleMatrix(1,1,mxREAL);
    out_nbe = mxGetPr(plhs[4]);  out_nbe[0] = nb_errors;  // nb of errors
    plhs[5] = mxCreateDoubleMatrix(1,1,mxREAL);
    out_eff = mxGetPr(plhs[5]);  out_eff[0] = last_npts_eff;  // nb of eff. pts used
    plhs[6] = mxCreateDoubleMatrix(1,1,mxREAL);
    out_nbw = mxGetPr(plhs[6]);  out_nbw[0] = last_nb_windows;  // nb of windows (for std computation)

    nlhs = 7;
    free(x); free(y); free(z);
    if (do_use_mask==1) free(mask);
}
