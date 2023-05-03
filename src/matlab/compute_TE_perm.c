/*
appelle la fonction  compute_transfer_entropy_symbolic() defined in entropy_sym.c

usage : compute_TE(x,y,[k]);

compilation sous Matlab avec la ligne suivante :
mex compute_TE.c entropy_sym.c math-tools.c -lgsl -lmwlapack -lmwblas -I./codeNG -I/opt/local/include/ -L./codeNG -L/opt/local/lib

*/


#include <math.h>
#include <stdio.h>
#include <string.h> // for memcpy
#include "mex.h"

#include "entropy_bins.h"
#include "entropy_perm.h"
#include "math_tools.h"  /* for the function "normalize" */

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

extern double compute_transfer_entropy_symbolic(double *x, double *y, int nx, int n_embed, int stride, int lag);


void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray  *prhs[])
{
  double *x, *y, *x_copy, *y_copy;
  int nx, n, n_embed=4, stride=1, lag=1;
  int size1,size2;
  double ret=0;
  double *xout1; /* returned values : estimates of TE */

  /* verify number of inputs */
  if (nrhs < 2) {
    mexErrMsgTxt("There are at least 2 input arguments required.\n");
    return;
  }  
  if (( nrhs > 3) && (nrhs < 5) ) {
    mexErrMsgTxt("You specified %d input parameters, this functions requires 2, 3 or 5 parameters. "
                                "See help for more information.\n");
    return;
  }


  /* first input the data */
  if(mxIsEmpty(prhs[0])) {
    mexErrMsgTxt("Input signal is empty!!\n");
    return;
  }
  if (!(mxIsDouble(prhs[0]))){ /* refuse complex data*/
    mexErrMsgTxt("Input argument must be of type double.");
  }	
  x=mxGetPr(prhs[0]);
  size1 = mxGetN(prhs[0]); 
  size2 = mxGetM(prhs[0]);

  if (size1 < size2){
    n=size1; nx=size2;
  } else {
    n=size2; nx=size1;
  }


  /* second input the data */
  if(mxIsEmpty(prhs[1])) {
    mexErrMsgTxt("Second input signal is empty!!\n");
    return;
  }
  if (!(mxIsDouble(prhs[1]))){ /* refuse complex data*/
    mexErrMsgTxt("Second input argument must be of type double.");
  }	
  y=mxGetPr(prhs[1]);
  size1 = mxGetN(prhs[1]); 
  size2 = mxGetM(prhs[1]);
  if (size1 < size2 && (size1 != n || size2 !=nx))
      mexErrMsgTxt("Input arguments must have the same size.");
  if (size2 < size1 && (size2 != n || size1 !=nx))
      mexErrMsgTxt("Input arguments must have the same size.");

  if (nrhs > 2) {
    /* third input number of neighbor */
    if(mxIsEmpty(prhs[2])) {
      mexErrMsgTxt("Input n_embed is empty!!\n");
    }
    n_embed=(int) mxGetScalar(prhs[2]);
    if (n_embed <= 0) mexErrMsgTxt("Input n_embed must be positive.\n");
  } else n_embed=4;

    if (nrhs > 3) 
    {   if(mxIsEmpty(prhs[3])) mexErrMsgTxt("Input stride is empty!!\n");
        stride = (int)mxGetScalar(prhs[3]);
        if (stride < 1) mexErrMsgTxt("stride must be larger or equal to 1.\n");
    
        if(mxIsEmpty(prhs[4])) mexErrMsgTxt("Input lag is empty!!\n");
        lag = (int)mxGetScalar(prhs[4]);
        if (lag < 1) mexErrMsgTxt("lag must be larger or equal to 1.\n");
    }

    /* returned values : estimate of MI by algo 1 and by algo 2 : */
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    xout1 = mxGetPr(plhs[0]);

    /* now we do maths : */
    x_copy = (double*)malloc(nx*sizeof(double)); memcpy(x_copy, x, nx*sizeof(double));
    y_copy = (double*)malloc(nx*sizeof(double)); memcpy(y_copy, y, nx*sizeof(double));
    normalize(x_copy, nx);
    normalize(y_copy, nx);
    
    ret = compute_transfer_entropy_symbolic(x_copy, y_copy, nx, n_embed, stride, lag);
    xout1[0] = ret;
    
    free(x_copy); free(y_copy);

}
