/*
tools to read parameters given to a Matlab function

N.B.G. 2021-12-22

2021-12-22 : new file, used by (almost) all matlab functions of the library
*/

#include <math.h>
#include <stdio.h>
#include <string.h> // for memcpy
#include "mex.h"
#include "matrix.h"

#include "entropy_ann.h"        // for non-masking version
#include "entropy_ann_mask.h"   // for masking version
#include "entropy_ann_threads.h"
#include "library_commons.h"    // for global variables

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf



void read_data_parameters(const mxArray *param, int *npts, int *mx, double **data_copy)
{   int size1, size2, npts_l, mx_l, i, j;
    double *data;
    
    if (mxIsEmpty(param)) mexErrMsgTxt("Input signal is empty!!\n");
    if (!(mxIsDouble(param))) mexErrMsgTxt("Input signal must be of type double.");

    size1 = mxGetN(param);
    size2 = mxGetM(param);
    if (size1<size2) {mx_l=size1; npts_l=size2;}
    else             {mx_l=size2; npts_l=size1;}
    
    if (*npts<0) *npts=npts_l; // typicaly first dataset to input defines npts
    else if (npts_l!=*npts) mexErrMsgTxt("Input signals must have the same size in time.");
    *mx=mx_l;
    
    // now we effectively read the data :
    data = mxGetPr(param);
    *data_copy = (double*)mxMalloc(npts_l*mx_l*sizeof(double));
    if (mxGetN(param) < mxGetM(param))  // we do NOT need to transpose
    {   memcpy(*data_copy, data, npts_l*mx_l*sizeof(double));
    }
    else // we need to transpose
    {   for (i=0; i<mx_l; i++)
        for (j=0; j<npts_l; j++) (*data_copy)[j + i*npts_l] = data[i + j*mx_l];
    }
} // end of function "read_data_parameters"



void read_embedding_parameters(const mxArray *param, int *px, int *py, int *pz, int *stride_x, int *stride_y, int *lag)
{   int i;
    double tmp;

    if (!mxIsStruct(param)) mexErrMsgTxt("argument is not a parameter struct!");
    for (i=0; i<mxGetNumberOfFields(param); i++)
    {
        if ( (px!=NULL) &&
               ( (strcmp(mxGetFieldNameByNumber(param,i),"mx")==0)
                || (strcmp(mxGetFieldNameByNumber(param,i),"m")==0)
                || (strcmp(mxGetFieldNameByNumber(param,i),"px")==0)
                || (strcmp(mxGetFieldNameByNumber(param,i),"p")==0)
                || (strcmp(mxGetFieldNameByNumber(param,i),"N")==0)) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *px = (int)tmp;
                if (*px < 1) mexErrMsgTxt("Input mx (embedding dim) must be larger or equal to 1.\n");
            }
        if ( (py!=NULL) &&
               ( (strcmp(mxGetFieldNameByNumber(param,i),"my")==0)
                || (strcmp(mxGetFieldNameByNumber(param,i),"py")==0) ) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *py = (int)tmp;
                if (*py < 1) mexErrMsgTxt("Input my (embedding dim) must be larger or equal to 1.\n");
            }
        if ( (pz!=NULL) &&
               ( (strcmp(mxGetFieldNameByNumber(param,i),"mz")==0)
                || (strcmp(mxGetFieldNameByNumber(param,i),"pz")==0) ) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *pz = (int)tmp;
                if (*pz < 1) mexErrMsgTxt("Input mz (embedding dim) must be larger or equal to 1.\n");
            }
        if ( (stride_x!=NULL) &&
                ( (strcmp(mxGetFieldNameByNumber(param,i),"stride")==0)
                || (strcmp(mxGetFieldNameByNumber(param,i),"stride_y")==0) 
                || (strcmp(mxGetFieldNameByNumber(param,i),"tau")==0) ) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *stride_x = (int)tmp;
                if (*stride_x < 1) mexErrMsgTxt("stride (tau) must be larger or equal to 1.\n");
            }
        if ( (stride_y!=NULL) &&
                ( (strcmp(mxGetFieldNameByNumber(param,i),"stride_y")==0) 
                || (strcmp(mxGetFieldNameByNumber(param,i),"tau_y")==0) ) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *stride_y = (int)tmp;
                if (*stride_y < 1) mexErrMsgTxt("stride (tau) for y must be larger or equal to 1.\n");
            }
        if ( (lag!=NULL) && (strcmp(mxGetFieldNameByNumber(param,i),"lag")==0) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                    tmp = *mxGetPr(p);
                    *lag = (int)tmp;
                    if (*lag < 1) mexErrMsgTxt("stride (tau) must be larger or equal to 1.\n");
            }
    }
} // end of function "read_embedding_parameters"



// 2022-05-24 : new function
void read_sampling_parameters(const mxArray *param, int *Theiler_x, int *Theiler_y, int *N_eff, int *N_real)
{   int i;
    double tmp;

    if (!mxIsStruct(param)) mexErrMsgTxt("argument is not a parameter struct!");
    for (i=0; i<mxGetNumberOfFields(param); i++)
        {
            if ( (Theiler_x!=NULL) && (strcmp(mxGetFieldNameByNumber(param,i),"Theiler_x")==0) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *Theiler_x = (int)tmp;
            }
            if ( (Theiler_y!=NULL) && (strcmp(mxGetFieldNameByNumber(param,i),"Theiler_y")==0) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *Theiler_y = (int)tmp;
            }
            if ( (N_eff!=NULL) && (strcmp(mxGetFieldNameByNumber(param,i),"N_eff")==0) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *N_eff = (int)tmp;
                if (*N_eff<-1) mexErrMsgTxt("N_eff must be positive, or -1 for automatic selection.\n");
                if (*N_eff==0) mexErrMsgTxt("N_eff cannot be 0!\n");            
            }
            if ( (N_real!=NULL) && 
                    ( (strcmp(mxGetFieldNameByNumber(param,i),"N_realizations")==0)
                    || (strcmp(mxGetFieldNameByNumber(param,i),"N_real")==0) ) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *N_real = (int)tmp;
                if (*N_real<-2) mexErrMsgTxt("N_real must be positive, or -1 or -2 for automatic selection.\n");
                if (*N_real==0) mexErrMsgTxt("N_real cannot be 0!\n");      
            }

        }
} // end of function "read_sampling_parameters"



// 2022-03-11 : added parameter "method" for functions that may use different algorithms
//              (e.g.: entropy rate)
void read_algorithms_parameters(const mxArray *param, int *k, int *algo, int *n_cores, int *method)
{   int i;
    double tmp;

    if (!mxIsStruct(param)) mexErrMsgTxt("argument is not a parameter struct!");
    for (i=0; i<mxGetNumberOfFields(param); i++)
        {
            if ( (k!=NULL) && (strcmp(mxGetFieldNameByNumber(param,i),"k")==0) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *k = (int)tmp;
                if (*k <= 0) mexErrMsgTxt("Input k must be positive.\n");
            }
            if ( (algo!=NULL) && (strcmp(mxGetFieldNameByNumber(param,i),"algo")==0) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *algo = (int)tmp;
                if ((*algo<1) || (*algo>3)) mexErrMsgTxt("Input algo must be either 1, 2 or 1+2.\n");
                ANN_choose_algorithm(*algo);
            }
            if ( (n_cores!=NULL) &&
                ( (strcmp(mxGetFieldNameByNumber(param,i),"threads")==0)
                || (strcmp(mxGetFieldNameByNumber(param,i),"cores")==0) ) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *n_cores = (int)tmp;
                if (*n_cores<0)  set_multithreading(2); // auto-adapt
                if (*n_cores==0) set_multithreading(0); // single-thread
                if (*n_cores>0)                         // multi-thread
                {   set_multithreading(1);
                    set_cores_number(*n_cores);
                }
            }
            if ( (n_cores!=NULL) && (strcmp(mxGetFieldNameByNumber(param,i),"method")==0) )
            {   mxArray *p = mxGetFieldByNumber(param,0,i);
                tmp = *mxGetPr(p);
                *method = (int)tmp;
//                if (*method <= 0) mexErrMsgTxt("Input method must be positive.\n");
            }

        }
} // end of function "read_algorithms_parameters"



void read_mask(const mxArray *param, char **mask, int *do_use_mask, int npts)
{   int size1, size2, j;
    double *m;

    if (mxIsEmpty(param)) mexErrMsgTxt("mask for epochs is empty!!\n");
    if ( mxIsDouble(param)
    /*        || mxIsClass(prhs[6], "char") || mxIsClass(prhs[6], "logical") || mxIsClass(prhs[6], "single")
            || mxIsClass(prhs[6], "int16")  || mxIsClass(prhs[6], "int32")  || mxIsClass(prhs[6], "int64")  || mxIsClass(prhs[6], "int8")
            || mxIsClass(prhs[6], "uint16") || mxIsClass(prhs[6], "uint32") || mxIsClass(prhs[6], "uint64") || mxIsClass(prhs[6], "uint8") */)
    {   m=mxGetPr(param);
        size1 = mxGetN(param); 
        size2 = mxGetM(param);
        if (size1 < size2) 
        {   if (size2!=npts) mexErrMsgTxt("Mask for epochs must have the same size (in time) as datasets.");
            if (size1!=1) mexErrMsgTxt("Mask for epochs must be a 1-dimensional vector");
        }
        else 
        {   if (size1!=npts) mexErrMsgTxt("Mask for epochs must have the same size (in time) as datasets.");
            if (size2!=1) mexErrMsgTxt("Mask for epochs must be a 1-dimensional vector");
        }
            
        *mask = (char*)malloc(npts*sizeof(char));
        for (j=0; j<npts; j++) (*mask)[j] = (char)m[j];
            
        *do_use_mask=1;
    }
    else mexErrMsgTxt("Mask for epochs must be of relevant type.\n");
} // end of function "read_mask"

