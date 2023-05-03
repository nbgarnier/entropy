/*
matlab wrapper for the function compute_entropy_rate_ann or its masked version
defined in entropy_ann_combinations.c (or entropy_ann_mask.c)

usage : h = compute_entropy_rate(x, [embed_params, [algo_params, [mask]]]) (see help)
 
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
    double *entropy, *out_std, *out_nbe, *out_eff, *out_nbw;
    int npts=-1, mx=1, px=1, stride=1;
    int Theiler=-4, N_eff=2048, N_real=10;
    int k=k_default, n_cores=-1, method=ENTROPY_RATE_MI;
//    int ret=0;
    int do_use_mask=0;
    char    *mask=NULL;

    /* verify number of inputs */
    if (nrhs < 1) mexErrMsgTxt("There is at least 1 input argument required. "
				"Please specify at least one vector x. "
				"See help for more information.\n");
    if (nrhs > 5) mexErrMsgTxt("Wrong number of input parameters, "
                "this function requires 1, 2, 3 or 4 parameters. "
				"See help for more information.\n");

    // input the dataset (first parameter) :
    read_data_parameters(prhs[0], &npts, &mx, &x);

    /* second input parameter contains embedding parameters : */
    if (nrhs > 1) read_embedding_parameters(prhs[1], &px, NULL, NULL, &stride, NULL, NULL);
    
   /* third input parameter contains sampling parameters : */
    if (nrhs > 2) read_sampling_parameters(prhs[2], &Theiler, NULL, &N_eff, &N_real);

    /* fourth input parameter contains algorithms parameters : */
    if (nrhs > 3) read_algorithms_parameters(prhs[3], &k, NULL, &n_cores, NULL);
  
    /* input parameter 5 : the mask (to select epochs) */
    if (nrhs==5) read_mask(prhs[4], &mask, &do_use_mask, npts);
    
    // output arguments / returned values :   
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    entropy = mxGetPr(plhs[0]);
        
    if (do_use_mask==0) compute_entropy_rate_ann     (x,       npts, mx, px, stride, Theiler, N_eff, N_real, k, method, entropy);
    else                compute_entropy_rate_ann_mask(x, mask, npts, mx, px, stride, k, method, entropy);

    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    out_std = mxGetPr(plhs[1]);  out_std[0] = last_std;  // std of the estimation
    plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
    out_nbe = mxGetPr(plhs[2]);  out_nbe[0] = nb_errors;  // nb of errors
    plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
    out_eff = mxGetPr(plhs[3]);  out_eff[0] = last_npts_eff;  // nb of eff. pts used
    plhs[4] = mxCreateDoubleMatrix(1,1,mxREAL);
    out_nbw = mxGetPr(plhs[4]);  out_nbw[0] = last_nb_windows;  // nb of windows (for std computation)
    
    nlhs = 5;
    
    mxFree(x);
    if (do_use_mask==1) free(mask);
}
