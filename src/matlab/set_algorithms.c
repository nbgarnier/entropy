/*
appelle la fonction ANN_choose_algorithm
defined in entropy_ANN.c
*/

#include <math.h>
#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
#include "mex.h"
#include "matrix.h"

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

#include "entropy_ann.h"

/*
// following code is to set a global varoiable in the Matlab global workspace
// https://fr.mathworks.com/help/matlab/matlab_external/set-and-get-variables-in-matlab-workspace.html
#include "mex.hpp"
#include "mexAdapter.hpp"

using matlab::mex::ArgumentList;
using namespace matlab::engine;
using namespace matlab::data;

class MexFunction : public matlab::mex::Function {
    ArrayFactory factory;
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
public:
    void operator()(ArgumentList outputs, ArgumentList inputs) {
        Array val = factory.createScalar(153.0);
        matlabPtr->setVariable(u"mexGlobal", val, WorkspaceType::GLOBAL);
    }
};

// better for C:
pointer = mexGetVariablePtr("MI_algo", "global")
mexPutVariable // http://www.ece.northwestern.edu/IT/local-apps/matlabhelp/techdoc/apiref/mexputvariable_01.html#1151514
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray  *prhs[])
{   double *tmp_return;
    int desired_algo;
    
    /* verify number of inputs */
    if (nrhs > 1) mexErrMsgTxt("This function takes 0 or 1 input argument. "
				"See help for more information.\n");
 		
    if (nrhs==1)
    {   desired_algo = (int)mxGetScalar(prhs[0]);
        ANN_choose_algorithm(desired_algo);
    }
    else
    {   ANN_choose_algorithm(-1);
    }
    
    // output argument / returned value :
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    tmp_return = mxGetPr(plhs[0]);
    tmp_return[0] = MI_algo;
    nlhs = 1;
    
    mexPutVariable("global", "MI_algo", plhs[0]);
}
