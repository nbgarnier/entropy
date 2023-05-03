/*
ApEn script/Mex for Matlab
2013-04-13 N.G.
2013-07-19 new wrapper or Matlab
2014-04-02 use new function (returns both ApEn and SampEn) -> forked outside in 'compute_KC.c'
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"

#include "math_tools.h"		// for the function "normalize"
#include "entropy_others.h"

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

extern double compute_ApEn_old  (double* data, int m, double r, int npts);
// extern int    compute_complexity(double* data, int m, double r, int npts, int kernel_type, double *ApEn, double *SampEn);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray  *prhs[])
{

  double *v_m;
  double *entropy;
  int nx, n, m=1; //, stride=1;
  int size1, size2;
  int i, j;
  double *x, r=0.5, std=1.0;
//  clock_t start, end; 
//  double cpu_time1,cpu_time2,cpu_time3;

  //start = clock();
  /* verify number of inputs */
  if (nrhs < 1) {
    mexErrMsgTxt("There is at least 1 input arguments required."
				"Please specify at least one vectors x. "
				"See help for more information.\n");
    return;
  }  
  if ( (nrhs != 4) ) {
    mexErrMsgTxt("Wrong number of input parameters, this function requires 4 parameters. "
				"See help for more information.\n");
    return;
  }


    /* input the data (1st parameter) */
    if(mxIsEmpty(prhs[0])) mexErrMsgTxt("Input signal is empty!!\n");
    if (!(mxIsDouble(prhs[0]))) mexErrMsgTxt("Input argument must be of type double."); /* refuse complex data*/

    v_m=mxGetPr(prhs[0]);
    size1 = mxGetN(prhs[0]); 
    size2 = mxGetM(prhs[0]);

    /* in case of nD we have to reorder the data */
    x=mxMalloc(size1*size2*sizeof(double));
    if (size1 < size2)
    {   n=size1; nx=size2;
        x=v_m;
    } 
    else 
    {   n=size2; nx=size1;
        m=0;
        for (i=0;i<n;i++)
        for (j=0;j<nx;j++)
        {   x[m]=v_m[(j*n)+i];
            m++;
        }
    }
    if (n!=1) mexErrMsgTxt("OK... you want to work on multidimensional data. This can be done, for sure,"
                " but it is not fully implemented here (by lack of interest...) contact NG.\n");
 
    if (nrhs > 1) 
    {   /* second input should be the (smaller) embedding dimension */
        if (mxIsEmpty(prhs[1])) mexErrMsgTxt("Input m is empty!!\n");
        m = (int)mxGetScalar(prhs[1]);
        if (m <= 0) mexErrMsgTxt("Input m must be positive.\n");
    } 
    else m=2;
  //end = clock();
  //cpu_time1 = ((double) (end - start)) / CLOCKS_PER_SEC;

 if (nrhs > 2) {
    /* 3rd and 4th input parameters */
    if(mxIsEmpty(prhs[2])) mexErrMsgTxt("Input r is empty!!\n");
    r  = mxGetScalar(prhs[2]);
    if (r <= 0) mexErrMsgTxt("Input r must be positive.\n");
	
	if(mxIsEmpty(prhs[3])) mexErrMsgTxt("Input std is empty!!\n");
    std = mxGetScalar(prhs[3]);
    if (std <= 0) mexErrMsgTxt("std must be positive.\n");
  }
  
    // output arguments / returned values :   
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    entropy = mxGetPr(plhs[0]);

    entropy[0] = compute_ApEn_old(x, m, r, nx); 
    nlhs =1;
  
}
