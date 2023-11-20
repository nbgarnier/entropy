/*
matlab wrapper for the function compute_entropy_ann_2d(x, nx, ny, p, stride, k, entropy) or its masked version
defined in entropy_ann.c (or entropy_ann_mask.c)

usage : S = compute_entropy_2d(x, [k, [m, stride, [mask]]]) (see help)
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for 'memcpy'
#include "mex.h"
#include "matrix.h"

// #include "math_tools.h"		// for the function "check_continuity"
#include "entropy_ann.h"
#include "library_commons.h"    // for global variables
#include "samplings.h"

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

#include "matlab_commons.c"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   double *v_m;
    double *entropy, *out_std, *out_nbe, *out_eff, *out_nbw;
    int nx, ny, mx=1, my=1, stride_x=1, stride_y=1;
    int Theiler_x=-4, Theiler_y=-4, N_eff=2048, N_real=10;
    int k=k_default, n_cores=-1;
    int d=1; // dimensionality of the data (code is OK if d>1)
    double  *x;
    mxArray *algo_type;

    algo_type=mexGetVariable("global", "MI_algo");
    if (algo_type==0) // couldn't get global value, using default
            MI_algo = MI_ALGO_1 | COUNTING_NG; // | MASK_OPTIMIZED;
    else    MI_algo = (int)mxGetScalar(algo_type);

    /* verify number of inputs */
    if (nrhs < 1) mexErrMsgTxt("There is at least 1 input argument required. Please specify at least one matrix or vector x.\n"
				"See help for more information.\n");
    if (nrhs > 4) mexErrMsgTxt("Wrong number of input parameters, this function requires 1, 2, 3 or 4 parameters.\n"
				"See help for more information.\n");

    /* input the data (1st parameter) */
    if(mxIsEmpty(prhs[0])) mexErrMsgTxt("Input signal is empty!!\n");
    if (!(mxIsDouble(prhs[0]))) mexErrMsgTxt("Input argument must be of type double."); /* refuse complex data*/
  
    v_m=mxGetPr(prhs[0]);
    nx = mxGetN(prhs[0]); // 2020-07-21: not sure of data ordering in matlab, should be OK
    ny = mxGetM(prhs[0]);

    x=mxMalloc(nx*ny*sizeof(double));
    memcpy(x, v_m, nx*ny*sizeof(double));

    /* second input parameter contains embedding parameters : */
    if (nrhs > 1) read_embedding_parameters(prhs[1], &mx, &my, NULL, &stride_x, &stride_y, NULL);
   
    /* third input parameter contains sampling parameters : */
    if (nrhs > 2) read_sampling_parameters(prhs[2], &Theiler_x, &Theiler_y, &N_eff, &N_real);

    /* fourth input parameter contains algorithms parameters : */
    if (nrhs > 3) read_algorithms_parameters(prhs[3], &k, NULL, &n_cores, NULL);
        
    // output arguments / returned values :   
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    entropy = mxGetPr(plhs[0]);
        
    compute_entropy_ann_2d(x, nx, ny, d, mx, my, stride_x, stride_y, Theiler_x, Theiler_y, N_eff, N_real, k, entropy);
    
    if (nlhs > 1) 
    {   plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_std = mxGetPr(plhs[1]);  out_std[0] = last_std;  // std of the estimation
    }
    if (nlhs > 2) 
    {   plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_nbe = mxGetPr(plhs[2]);  out_nbe[0] = nb_errors;  // nb of errors
    }
    if (nlhs > 3) 
    {   plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_eff = mxGetPr(plhs[3]);  out_eff[0] = last_samp.N_eff;  // nb of eff. pts used
    }
    if (nlhs > 4) 
    {   plhs[4] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_nbw = mxGetPr(plhs[4]);  out_nbw[0] = last_samp.N_real;  // nb of windows (for std computation)
    }
//    nlhs = 5;
    mxFree(x);
}
