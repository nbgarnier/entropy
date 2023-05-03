/*
appelle la fonction switch_ANN_ALLOW_SELF_MATCH
defined in ANN_wrapper.c

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

//#include "ANN_wrapper.h"
#ifdef __cplusplus
extern "C" {
#endif 
char set_ANN_state(int my_choice);
char get_ANN_state(void);
#ifdef __cplusplus
}
#endif 

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray  *prhs[])
{   double *tmp_return;
    int ret;
    char    ANN_SELF_COUNT = 10;

    /* verify number of inputs */
    if (nrhs > 1) mexErrMsgTxt("There is 0 or 1 input argument required. "
				"See help for more information.\n");
 		
    /* input parameter 1 : the option to change ANN settings to self-count center points */
    if (nrhs==1) 
    {   ANN_SELF_COUNT = (int)mxGetScalar(prhs[0]);
        ret = set_ANN_state(ANN_SELF_COUNT);
        printf("new : %d\n",ret);
    }   
    else 
    {   ret = get_ANN_state();
        printf("previous : %d\n",ret);    
    } 
    
    // output arguments / returned values :   
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    tmp_return = mxGetPr(plhs[0]);
    tmp_return[0] = ret;
    nlhs = 1;

}
