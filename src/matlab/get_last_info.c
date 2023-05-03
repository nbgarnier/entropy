/*
returns some informations on the last computation of the library

usage : [] = get_last_info(0)
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for 'memcpy'
#include "mex.h"
#include "matrix.h"

#include "entropy_ann.h"
#include "library_commons.h" // for global variables

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   double *v_m;
    double *output;
//    mxArray *algo_type;
    int verbosity=0;

/*    algo_type=mexGetVariable("global", "MI_algo");
    if (algo_type==0) // couldn't get global value, using default
            MI_algo = MI_ALGO_1 | COUNTING_NG | MASK_OPTIMIZED;
    else    MI_algo = (int)mxGetScalar(algo_type);
*/

    /* verify number of inputs */
    if (nrhs < 1) verbosity=0; // default value
    if (nrhs > 1) mexErrMsgTxt("Wrong number of input parameters, "
                "this function requires 0 or 1 parameters. "
				"See help for more information.\n");

    /* input the data (1st parameter) */
    if (mxIsEmpty(prhs[0])) mexErrMsgTxt("first parameter is empty!!\n");
    if (!(mxIsDouble(prhs[0]))) mexErrMsgTxt("Input argument must be of type double."); /* refuse complex data*/
  
    v_m=mxGetPr(prhs[0]);
    verbosity = (int)v_m[0];

    // output arguments / returned values :   
    plhs[0] = mxCreateDoubleMatrix(1,4,mxREAL);
    output  = mxGetPr(plhs[0]);
    output[0] = last_std;
    output[1] = last_std2;
    output[2] = nb_errors;
    output[3] = last_npts_eff;
    output[4] = last_nb_windows;
    nlhs = 1;
        
    if (verbosity>0) 
    {   printf("from last function call:\n");
        printf("- standard deviation(s):       %f, %f\n", last_std, last_std2);
        printf("- nb of errors encountered:    %d\n", nb_errors);
        printf("- effective number of points:  %d\n", last_npts_eff);
        printf("- nb of independent windows:   %d\n", last_nb_windows);
    }   

}
