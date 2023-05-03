/*
normalize_nd : for normalizing a multivariate dataset
*/

#include <math.h>
#include <stdio.h>
#include <string.h> // for memcpy
#include <stdlib.h>
//#include <time.h>
#include "mex.h"
//#include "matrix.h"

#include "math_tools.h"		// for the function "normalize"

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

extern double normalize_nd     (double *x,             int m, int npts);
extern double normalize_nd_mask(double *x, char *mask, int m, int npts);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray  *prhs[])
{   int npts, mx;
    int size1, size2;
    register int i, j;
    double *x, *x_copy, *det_cov, *m;
    char    *mask, do_use_mask=0;

    /* verify number of inputs */
    if ((nrhs<1) || (nrhs>2)) mexErrMsgTxt("This function requires 1 or 2 input arguments.\n"
				"See help for more information.\n");
  
    /* input the first parameter = the data */
    if (mxIsEmpty(prhs[0]))     mexErrMsgTxt("Input signal is empty!!\n");
    if (!(mxIsDouble(prhs[0]))) mexErrMsgTxt("Input argument must be of type double.");
    
    x=mxGetPr(prhs[0]);
    size1 = mxGetN(prhs[0]); 
    size2 = mxGetM(prhs[0]);

    if (size1 < size2)  { mx=size1; npts=size2; } 
    else				{ mx=size2; npts=size1; }
    
    /* input the second paramter = the mask (to select epochs) */
    if (nrhs==2) 
    {   if (mxIsEmpty(prhs[1])) mexErrMsgTxt("mask for epochs (parameter 2) is empty!!\n");
        if ( mxIsDouble(prhs[1])
    /*        || mxIsClass(prhs[6], "char") || mxIsClass(prhs[6], "logical") || mxIsClass(prhs[6], "single")
            || mxIsClass(prhs[6], "int16")  || mxIsClass(prhs[6], "int32")  || mxIsClass(prhs[6], "int64")  || mxIsClass(prhs[6], "int8")
            || mxIsClass(prhs[6], "uint16") || mxIsClass(prhs[6], "uint32") || mxIsClass(prhs[6], "uint64") || mxIsClass(prhs[6], "uint8") */)
        {   m=mxGetPr(prhs[1]);
            size1 = mxGetN(prhs[1]); 
            size2 = mxGetM(prhs[1]);
            if (size1 < size2) 
            {   if (size2!=npts) mexErrMsgTxt("Mask for epochs must have the same size (in time) as dataset.");
                if (size1!=1) mexErrMsgTxt("Mask for epochs must be a 1-dimensional vector");
            }
            else 
            {   if (size1!=npts) mexErrMsgTxt("Mask for epochs must have the same size (in time) as dataset.");
                if (size2!=1) mexErrMsgTxt("Mask for epochs must be a 1-dimensional vector");
            }
            
            mask = (char*)malloc(npts*sizeof(char)); 
            for (j=0; j<npts; j++) mask[j] = (char)m[j];
            
            do_use_mask=1;
        }
        else mexErrMsgTxt("Mask for epochs (parameter 2) must be of relevant type.\n");
    }
  
    // output arguments / returned values :   
    plhs[0] = mxCreateDoubleMatrix(mx, npts, mxREAL);
    x_copy  = mxGetData(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    det_cov = mxGetPr(plhs[1]);

    if (mxGetN(prhs[0]) < mxGetM(prhs[0]))  // we do NOT need to transpose
    {   //printf("%f %f %f...\n", x[0], x[1], x[2]);
        memcpy(x_copy, x, npts*mx*sizeof(double));
    }
    else // we need to transpose
    {   //printf("%f %f %f...\n", x[0], x[mx], x[2*mx]);
        for (i=0; i<mx; i++) 
        for (j=0; j<npts; j++)
        {   x_copy[j + i*npts] = x[i + j*mx];
        }
    }

    if (do_use_mask==0)    det_cov[0] = normalize_nd     (x_copy,       mx, npts);
    else                   det_cov[0] = normalize_nd_mask(x_copy, mask, mx, npts);
    
    if (do_use_mask==1) free(mask);
}
