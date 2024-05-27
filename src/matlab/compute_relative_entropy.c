/*
wrapper for the function "compute_relative_entropy_ann" defined in entropy_ann.c

 % [Hr, H, KL] = compute_relative_entropy(x, y, [embed_params, [algo_params, [mask]]])
 
2021-12-23 NBG
*/

#include <stdio.h>
#include <string.h> // for 'memcpy'
#include "mex.h"

#include "entropy_ann.h"        // for non-masking version
#include "entropy_ann_mask.h"
#include "entropy_ann_threads.h"
#include "library_commons.h"    // for global variables
#include "samplings.h"

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

#include "matlab_commons.c"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray  *prhs[])
{   double  *x,*y;
    double  *xout1, *xout2, *xout3, *out_std, *out_nbe, *out_eff, *out_nbw;
    char    *mask;
    int     npts_x=-1, npts_y=-1, mx=1, my=1, px=1, py=1, stride=1;
    int     Theiler=-4, N_eff=2048, N_real=10;
    int     k=k_default, n_cores=-1;
    int     ret1, do_use_mask=0;    
    /* verify the number of input parameters */
    if (nrhs < 2) mexErrMsgTxt("There are at least 2 input arguments required. Please specify at least two vectors x and y.\n"
                    "See help for more information.\n");
    if (nrhs > 5) mexErrMsgTxt("You specified %d input parameters, this functions requires 2, 3, 4 or 6 parameters.\n"
                    "See help for more information.\n");
    if (nlhs < 3) mexErrMsgTxt("There are at least 3 output values. Please use at least 3 output variables.\n"
                    "See help for more information.\n");


    // parameters 1 and 2 : datasets
    read_data_parameters(prhs[0], &npts_x, &mx, &x);
    read_data_parameters(prhs[1], &npts_y, &my, &y);
//    printf("x has dimension %d and %d in time, whereas y has dimension %d and %d in time\n", mx, npts, my, npts);

    /* parameter 3 contains embedding parameters : */
    if (nrhs > 2) read_embedding_parameters(prhs[2], &px, &py, NULL, &stride, NULL, NULL);

    /* third input parameter contains sampling parameters : */
    if (nrhs > 3) read_sampling_parameters(prhs[2], &Theiler, NULL, &N_eff, &N_real);

    /*  parameter 4 contains algorithms parameters : */
    if (nrhs > 4) read_algorithms_parameters(prhs[3], &k, NULL, &n_cores, NULL);

    /* parameter 5 contains the mask (to select epochs) */
    if (nrhs==6) read_mask(prhs[4], &mask, &do_use_mask, npts_x);
    if ( (do_use_mask>0)  )
        mexErrMsgTxt("sorry, this function does not support masking yet. Contact nicolas.garnier@ens-lyon.fr if interested.\n");
    
    // test on dimensions :
    if (mx*px != my*py) mexErrMsgTxt("total dimension of x (mx*px) and y (my*py) are different!\n");

    /* returned values : estimate of relative entropy, entropy, and KL divergence: */
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL); // relative entropy
    xout1 = mxGetPr(plhs[0]);

    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL); // entropy
    xout2 = mxGetPr(plhs[1]);
     
    plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL); // KL divergence
    xout3 = mxGetPr(plhs[2]);

//    ret1  = compute_relative_entropy_ann(x, npts_x, y, npts_y, mx, my, px, py, stride, k, xout1);
    ret1  = compute_relative_entropy_ann(x, npts_x, y, npts_y, mx, my, px, py, stride, Theiler, N_eff, N_real, k, xout1);
//    ret1 += compute_entropy_ann         (x, npts_x, mx, px, stride, k, xout2);
    ret1 += compute_entropy_ann         (x, npts_x, mx, px, stride, Theiler, N_eff, N_real, k, xout2);
    if (ret1!=0) printf("[Warning] return value code %d\n", ret1);
    *xout3 = *xout1 - *xout2;

    if (nlhs>3)
    {   plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_std = mxGetPr(plhs[3]);  out_std[0] = last_std;  // std of the estimation
    }
    if (nlhs>4)
    {   plhs[4] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_nbe = mxGetPr(plhs[4]);  out_nbe[0] = nb_errors;  // nb of errors
    }
    if (nlhs>5)
    {   plhs[5] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_eff = mxGetPr(plhs[5]);  out_eff[0] = last_samp.N_eff;  // nb of eff. pts used
    }
    if (nlhs>6)
    {   plhs[6] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_nbw = mxGetPr(plhs[6]);  out_nbw[0] = last_samp.N_real;  // nb of windows (for std computation)
    }
    
    free(x); free(y);
    if (do_use_mask) free(mask);
}
