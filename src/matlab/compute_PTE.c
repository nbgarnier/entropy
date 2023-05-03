/* to compute partial trnsfer entropy

usage : see .m script in bin/matlab

// 2013-07-19 : fork from compute_PMI
// 2021-12-21 : new parameters convention
*/

#include <math.h>
#include <stdio.h>
#include <string.h> // for memcpy
#include "mex.h"

#include "entropy_ann.h"
#include "entropy_ann_mask.h"
#include "entropy_ann_threads.h"
#include "library_commons.h"    // for global variables


#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

#include "matlab_commons.c"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray  *prhs[])
{   double  *x, *y, *z;
    double  *xout1, *xout2, *out_std1, *out_std2, *out_nbe, *out_eff, *out_nbw;
    char    *mask=NULL;
    int     npts=-1, dim[6], stride=1, lag=1;
    int     Theiler=-4, N_eff=2048, N_real=10;
    int     k=k_default, algo=1, n_cores=-1;
//    int     ret=0; /* value returned by C function; == 0 if OK */
    int     do_use_mask=0;
    
    /* verify the number of input parameters */
    if (nrhs < 3) mexErrMsgTxt("There are at least 3 input arguments required. "
				"Please specify at least 3 vectors x, y and z. "
				"See help for more information.\n");
    if (nrhs > 7) mexErrMsgTxt("You specified %d input parameters, this functions requires 3, 4, 5, 6 or 7 parameters. "
                               "See help for more information.\n");

    // input the 3 datasets (3 first parameters) :
    read_data_parameters(prhs[0], &npts, &dim[0], &x);
    read_data_parameters(prhs[1], &npts, &dim[1], &y);
    read_data_parameters(prhs[2], &npts, &dim[2], &z);

    /* parameter 4 contains embedding parameters : */
    if (nrhs > 3) read_embedding_parameters(prhs[3], &dim[3+0], &dim[3+1], &dim[3+2], &stride, NULL, &lag);

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

    if (do_use_mask==0)
    {   mask = (char*)malloc(npts*sizeof(char)); 
        for (int j=0; j<npts; j++) mask[j] = (char)1;   // fake mask
    }
 
    compute_partial_TE_ann_mask(x,y,z, mask, npts, dim, stride, lag, k, xout1, xout2);

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
    free(mask);
}
