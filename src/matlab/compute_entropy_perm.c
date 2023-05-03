/*
compute permutation entropy
Matlab version

this Matlab wrapper is a fork from compute_entorpy.c
2012-06-06 : first version

*/

/*  pour acceder a la transposer de a
    #define mat(a, i, j) (*(a + (m*(j)+i))) 
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"

#include "math_tools.h"		// for the function "normalize"
#include "entropy_bins.h"
#include "entropy_perm.h"

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

extern double compute_entropy_symbolic(double *x, int nx, int m, int n_embed, int stride);


void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray  *prhs[])
{   double *v_m;
    double *out;
    int nx, n, n_embed=2, stride=1;
    int size1, size2;
    int i, j, m;
    double *x;

    /* verify number of inputs */
    if (nrhs < 1) {
        mexErrMsgTxt("There is at least 1 input arguments required."
				"Please specify at least one vectors x. "
				"See help for more information.\n");
        return;
    }  
    if (nrhs > 3)
    {   mexErrMsgTxt("Wrong number of input parameters, this function requires 1, or 2, or 3 parameters. "
				"See help for more information.\n");
        return;
    }

    /* first input the data */
    if(mxIsEmpty(prhs[0]))
    {   mexErrMsgTxt("Input signal is empty!!\n");
        return;
    }
    if (!(mxIsDouble(prhs[0])))/* refuse complex data*/
    {   mexErrMsgTxt("Input argument must be of type double.");
        return;
    }	
    v_m   = mxGetPr(prhs[0]);
    size1 = mxGetN(prhs[0]); 
    size2 = mxGetM(prhs[0]);

    /*mexPrintf("size1=%d size2=%d\n",size1,size2);*/
    /* in case of nD we have to reorder the data */
    x=mxMalloc(size1*size2*sizeof(double));
    if (size1 < size2)
    {   n=size1; nx=size2;
        x=v_m;
/*       for (i=0;i<n*nx;i++) x[i]=v_m[i]; */
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

  /*mexPrintf("nx=%d n=%d\n",nx,n);*/
  
    if (nrhs > 1) /* then number m of embedding dimensions */
    {   if(mxIsEmpty(prhs[1])) mexErrMsgTxt("Input m is empty!!\n");
        n_embed = (int)mxGetScalar(prhs[1]);
        if (n_embed < 1) mexErrMsgTxt("Input parameter m (embedding size) must be larger or equal to 1.\n");
	}
    
	if (nrhs > 2) /* then number m of embedding dimensions */
    {   if(mxIsEmpty(prhs[2])) mexErrMsgTxt("Input stride is empty!!\n");
        stride = (int)mxGetScalar(prhs[2]);
        if (stride < 1) mexErrMsgTxt("Input parameter stride must be larger or equal to 1.\n");
    }
    else stride=1;
  
    // output arguments / returned values :   
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    out     = mxGetPr(plhs[0]);

    // now, the math part : 
    out[0]  = compute_entropy_symbolic(x, nx, n, n_embed, stride);

}
