/*
matlab wrapper for the fonction compute_directed_information_ann() defined in entropy_ann.c
or its masked version

 usage : see .m in /bin/matlab/

 2013-07-19 : masking version
 2021-12-21 : multi-thread and new parameters conventions

*/

#include <math.h>
#include <stdio.h>
#include <string.h> // for memcpy
#include "mex.h"

#include "entropy_ann_combinations.h"   // for the function compute_DI_ann
#include "entropy_ann_mask.h"           // for the function compute_DI_ann_mask
#include "entropy_ann_threads.h"
#include "library_commons.h"            // for global variables
#include "samplings.h"

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

#include "matlab_commons.c"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray  *prhs[])
{   double  *x, *y;
    double  *xout1, *xout2, *out_std1, *out_std2, *out_nbe, *out_eff, *out_nbw;
    int     npts=-1, mx, my, N, stride=1;
    int     Theiler=-4, N_eff=2048, N_real=10;
    int     k=k_default, algo=1, n_cores=-1;
//    int     ret=0;
    char    *mask=NULL;
    int     do_use_mask=0;
    

    /* verify number of input parameters */
    if (nrhs < 2) mexErrMsgTxt("There are at least 2 input arguments required.\n"
                                "Please specify at least 2 vectors or matrices x, y.\n"
                               "See help for more information.\n");
    if (nrhs > 5) mexErrMsgTxt("You specified %d input parameters, this functions requires 2, 3, 4 or 5 parameters. "
                               "See help for more information.\n");
   
    // input the 2 datasets (2 first parameters) :
    printf("npts=%d\n", npts);
    read_data_parameters(prhs[0], &npts, &mx, &x); printf("npts=%d\n", npts);
    read_data_parameters(prhs[1], &npts, &my, &y); printf("npts=%d\n", npts);
    
    /* parameter 3 contains embedding parameters : */
    if (nrhs>2) read_embedding_parameters(prhs[2], &N, NULL, NULL, &stride, NULL, NULL);

    /* parameter 4 contains sampling parameters : */
    if (nrhs > 3) read_sampling_parameters(prhs[3], &Theiler, NULL, &N_eff, &N_real);

    /* parameter 5 contains algorithms parameters : */
    if (nrhs > 4) read_algorithms_parameters(prhs[4], &k, &algo, &n_cores, NULL);

    /* parameter 6 contains the mask (to select epochs) */
    if (nrhs==6) read_mask(prhs[5], &mask, &do_use_mask, npts);
	
    /* returned values : estimate of MI by algo 1 and by algo 2 : */
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    xout1 = mxGetPr(plhs[0]);

    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    xout2 = mxGetPr(plhs[1]);


    if (do_use_mask==1) compute_directed_information_ann_mask(x,y, mask, npts, mx, my, N, stride, Theiler, N_eff, N_real, k, xout1, xout2);
    else                compute_directed_information_ann     (x,y,       npts, mx, my, N, stride, Theiler, N_eff, N_real, k, xout1, xout2);
    
    if (nlhs>2) 
    {   plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_std1 = mxGetPr(plhs[2]);  out_std1[0] = last_std;  // std of the estimation
    }
    if (nlhs>3) 
    {   plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_std2 = mxGetPr(plhs[3]);  out_std2[0] = last_std2;  // std of the estimation
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
    if (do_use_mask==1) free(mask);
}
