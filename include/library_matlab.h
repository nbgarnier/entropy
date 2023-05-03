/*
 *  library_matlab.h
 *  
 *
 *  Created by Nicolas Garnier on 2021-12-21.
 *  Copyright 2012-2021 ENS-Lyon - CNRS. All rights reserved.
 *
 *  
 */

#ifndef LIBRARY_MATLAB
#define LIBRARY_MATLAB

#ifdef MATLAB_MEX_FILE  /* si la compilation est demandée par Matlab */
#include <mex.h>
#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf
#else				 /* si la compilation n'est pas demandée par Matlab */
#include <stdlib.h>  /* for malloc, NULL, etc */
#include <stdio.h>   /* for printf */
#endif 

#endif
