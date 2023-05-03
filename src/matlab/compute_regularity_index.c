/*
matlab wrapper for the function compute_regularity_index_ann or its masked version
defined in entropy_ann_combinations.c 

usage : h = compute_regularity_index(x, [embed_params, [algo_params, [mask]]]) (see help)
 
// 2022-03-11, new parameters convention
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>             // for 'memcpy' and 'strcmp'
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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   double *x;
    double *I1, *I2, *out_std1, *out_std2, *out_nbe, *out_eff, *out_nbw;
    int npts=-1, mx=1, px=1, stride=1;
    int k=k_default, n_cores=-1;
    int do_use_mask=0;
    char    *mask=NULL;

    /* verify number of inputs */
    if (nrhs < 1) mexErrMsgTxt("There is at least 1 input argument required. "
				"Please specify at least one vector x. "
				"See help for more information.\n");
    if (nrhs > 3) mexErrMsgTxt("Wrong number of input parameters, "
                "this function requires 1, 2, or 3 parameters. "
				"See help for more information.\n");

    // input the dataset (first parameter) :
    read_data_parameters(prhs[0], &npts, &mx, &x);

    /* second input parameter contains embedding parameters : */
    if (nrhs > 1) read_embedding_parameters(prhs[1], &px, NULL, NULL, &stride, NULL,  NULL);
    
    /* third input parameter contains algorithms parameters : */
    if (nrhs > 2) read_algorithms_parameters(prhs[2], &k, NULL, &n_cores, NULL);
  
    /* input parameter 4 : the mask (to select epochs) */
    if (nrhs==4) read_mask(prhs[3], &mask, &do_use_mask, npts);
    
    // returned values : estimate by algo 1 and by algo 2 : */
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    I1 = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    I2 = mxGetPr(plhs[1]);

    if (do_use_mask==0) compute_regularity_index_ann(x,       npts, mx, px, stride, k, I1, I2);
    else mexErrMsgTxt("not coded yet"); // masked version not coded yet

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

    mxFree(x);
    if (do_use_mask==1) free(mask);
}
