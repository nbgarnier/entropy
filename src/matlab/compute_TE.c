/*
 matlab wrapper for the function compute_transfert_entropy_nd_ann()      defined in entropy_ann.c
           or its masked version compute_transfert_entropy_nd_ann_mask() defined in entropy_ann_mask.c

 usage : see .m in /bin/matlab/

 2013-07-19 : masking version
 2021-12-21 : multi-thread and new parameters conventions

*/

#include <math.h>
#include <stdio.h>
#include <string.h> // for memcpy
#include "mex.h"

#include "entropy_ann_combinations.h"
#include "entropy_ann_mask.h"
#include "entropy_ann_threads.h"
#include "library_commons.h"    // for global variables

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

#include "matlab_commons.c"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray  *prhs[])
{   double *x=NULL, *y=NULL;
    double *xout1, *xout2, *out_std1, *out_std2, *out_nbe, *out_eff, *out_nbw;
    char  *mask=NULL;
    int   npts=-1, mx=1, my=1, px=1, py=1, stride=1, lag=1;
    int     Theiler=-4, N_eff=2048, N_real=10;
    int   k=k_default, algo=1, n_cores=-1;
//    int   ret=0;
    int do_use_mask=0;
    
    /* verify number of input parameters */
    if (nrhs < 2) mexErrMsgTxt("There are at least 2 input arguments required. Please specify at least 2 vectors or matrices x, y.\n"
                               "See help for more information.\n");
    if (nrhs > 6) mexErrMsgTxt("You specified %d input parameters, this functions requires 2, 3, 4, 5 or 6 parameters.\n"
                               "See help for more information.\n");
    
    // input the 2 datasets (2 first parameters) :
    read_data_parameters(prhs[0], &npts, &mx, &x);
    read_data_parameters(prhs[1], &npts, &my, &y);
    
    /* parameter 3 contains embedding parameters : */
    if (nrhs>2) read_embedding_parameters(prhs[2], &px, &py, NULL, &stride, NULL, &lag);

    /* parameter 4 contains sampling parameters : */
    if (nrhs > 3) read_sampling_parameters(prhs[3], &Theiler, NULL, &N_eff, &N_real);

    /*  parameter 5 contains algorithms parameters : */
    if (nrhs > 4) read_algorithms_parameters(prhs[4], &k, &algo, &n_cores, NULL);

    /* parameter 6 contains the mask (to select epochs) */
    if (nrhs==6) read_mask(prhs[5], &mask, &do_use_mask, npts);
	
    /* returned values : estimate of TE by algo 1 and by algo 2 : */
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    xout1 = mxGetPr(plhs[0]);

    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    xout2 = mxGetPr(plhs[1]);
    
    if (do_use_mask==1) compute_transfer_entropy_ann_mask(x, y, mask, npts, mx, my, px, py, stride, lag, k, xout1, xout2);
    else                compute_transfer_entropy_ann     (x, y,       npts, mx, my, px, py, stride, lag, k, xout1, xout2);
    
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
    free(x); free(y);
    if (do_use_mask==1) free(mask);
}
