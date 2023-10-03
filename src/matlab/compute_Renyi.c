/* to compute the Renyi entropy of order q of any signal

// 2021-12-21, new parameters convention
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"

#include "entropy_ann_Renyi.h"
#include "entropy_ann_mask.h"
#include "entropy_ann_threads.h"
#include "library_commons.h"    // for global variables
#include "samplings.h"

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

#include "matlab_commons.c"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray  *prhs[])
{   double *x;
    double *entropy, *out_std, *out_nbe, *out_eff, *out_nbw;
    int     npts=-1, mx=1, px=1, stride=1;
    int     Theiler=-4, N_eff=2048, N_real=10;
    int     k=k_default, n_cores=-1;
    int     method=0;   // 0 for embedding, 1 for increments
    int     do_use_mask=0;
    double  q=2;
    char    *mask=NULL;

    /* verify number of inputs */
    if (nrhs < 2) mexErrMsgTxt("There are at least 2 input arguments required. "
				"Please specify at least one vector x and one value of order q. "
				"See help for more information.\n");
    if (nrhs > 6) mexErrMsgTxt("Wrong number of input parameters, "
                "this function requires 1, 2, 3, 4, 5 or 6 parameters. "
                "See help for more information.\n");
    
    // input the dataset (first parameter) :
    read_data_parameters(prhs[0], &npts, &mx, &x);

    /* second input parameter is the order q of the Renyi entropy : */
    if (mxIsEmpty(prhs[1])) mexErrMsgTxt("Input q is empty!!\n");
    q = mxGetScalar(prhs[1]);
    if (q == 1) mexErrMsgTxt("Input q must be different from 1.\n");
    
    /* third input parameter contains embedding parameters : */
    if (nrhs > 2) read_embedding_parameters(prhs[2], &px, NULL, NULL, &stride, NULL, NULL);

    /* 4th input parameter contains sampling parameters : */
    if (nrhs > 3) read_sampling_parameters(prhs[3], &Theiler, NULL, &N_eff, &N_real);

    /* 5th input parameter contains algorithms parameters : */
    if (nrhs > 4) read_algorithms_parameters(prhs[4], &k, NULL, &n_cores, NULL);
  
    /* input parameter 6 : the mask (to select epochs) */
    if (nrhs==6) read_mask(prhs[5], &mask, &do_use_mask, npts);
	
    // output arguments / returned values :   
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    entropy = mxGetPr(plhs[0]);
 
    if (do_use_mask==0) compute_Renyi_ann     (x,       npts, mx, px, stride, q, Theiler, N_eff, N_real, k, method, entropy);
    else                compute_Renyi_ann_mask(x, mask, npts, mx, px, stride, q, Theiler, N_eff, N_real, k, method, entropy);
    // ret contains the number of errors encountered
    if (nlhs>1)
    {   plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_std = mxGetPr(plhs[1]);  out_std[0] = last_std;         // std of the estimation
    }
    if (nlhs>2)
    {   plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_nbe = mxGetPr(plhs[2]);  out_nbe[0] = nb_errors;        // nb of errors
    }
    if (nlhs>3)
    {   plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_eff = mxGetPr(plhs[3]);  out_eff[0] = last_samp.N_eff;    // effective nb of pts used
    }
    if (nlhs>4)
    {   plhs[4] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_nbw = mxGetPr(plhs[4]);  out_nbw[0] = last_samp.N_real;  // nb of windows (for std computation)
    }
    nlhs = 5;
    
    mxFree(x);
    if (do_use_mask==1) free(mask);
}



