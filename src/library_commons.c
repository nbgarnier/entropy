/*
 * library_commons.c
 *
 *  Created by Nicolas Garnier on 2020-03-05.
 *  Copyright 2012-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 *
 */
#include <math.h>       // for fabs
#include <float.h>      // for DBL_MIN (2020-07-17)
#include <stdio.h>      // for printf

#include "library_commons.h" // for definitions of nb_errors, and stds

// global variables that changes the behavior of the library
int lib_verbosity=1;        // <0 : no messages, even if an error is encountered (not recommended!)
                            // 0  : messages only if an error is encountered (default)
                            // 1  : important warnings only
                            // 2 or more: more and more warnings
int lib_warning_level=1;    // 0 : no physical checks, code trusts user / user is responsible
                            // 1  : physical checks raises warnings 
                            // 2  : physical checks raises errors 
                            //      (1 or 2 helps the user, but limits advanced use of the library)

// global variables to count errors encountered in functions
int nb_errors_local=0;      // errors encountered in an "engine" function 
                            // (to be used inside "wrapper" functions)
int nb_errors=0;            // errors encountered in a "wrapper" function (usually returned)
int nb_errors_total=0;      // total nb of errors encountered since the library has been loaded
                            // (Cython only ?) (unused)

// global variables to count effective number of points used in functions
int last_npts_eff=0;        // effective nb of points used in the last "wrapper" function call
int last_npts_eff_local=0;  // effective nb of points used in the last "engine" function call

// global variable to keep track of the standard deviation of the last estimate
double last_std=my_NAN;     // main variable
double last_std2=my_NAN;    // additional variable (for functions returning a second value)

// global variables to record the std (and its std) of the pre-processed data
// (added 2022-03-11, to keep track of basic stats of the input data)
// (useful for increments)
double data_std=my_NAN;     // std of the data (e.g., increments, which can be intermediate)
double data_std_std=my_NAN; // std of the std of the data



/****************************************************************************************/
/* selects the verbosity level                                                          */
/****************************************************************************************/
void ANN_set_verbosity(int level)
{   lib_verbosity=level;
    return;
}


/****************************************************************************************/
/* selects the algorithms to use:                                                       */
/* - either algorithm 1 or 2 from Kraskov-Stogbauer-Grassberger (2004)                  */
/* - either old (legacy, heavily used) counting or new one (with ANN and extra trees)   */
/*                                                                                      */
/* defaults values are algorithm 1 and legacy counting                                  */
/*                                                                                      */
/* these choices propagates automatically in the Cython version                         */
/*                                                                                      */
/* 2019-01-28 : first version with KSG algos choice only                                */
/* 2019-01-30 : added legacy/ANN counting choice                                        */
/* 2019-02-05 : added parameter for masks (optimized or conservative)                   */
/****************************************************************************************/
void ANN_choose_algorithm(int algo)
{
    if (algo>0) // then we have to set the algorithms
    {   MI_algo = 0; // a welcome reset
        if (algo&MI_ALGO_1)             MI_algo |= MI_ALGO_1;
        if (algo&MI_ALGO_2)             MI_algo |= MI_ALGO_2;
        // check that at least one algo is selected:
        if (!(MI_algo&MI_ALGO_1) && !(MI_algo&MI_ALGO_2)) MI_algo |= MI_ALGO_1;
  
        if (algo&COUNTING_ANN)          MI_algo |= COUNTING_ANN;
        else                            MI_algo |= COUNTING_NG;
        
        if (algo&COUNTING_TEST)         MI_algo |= COUNTING_TEST;
    }
    else // otherwise, we read and print the (previous) algorithms parameters:
    {   
        printf("   -> Kraskov, Stogbauer, Grassberger algorithm ");
        if (MI_algo&MI_ALGO_1) { printf("1 "); if (MI_algo&MI_ALGO_2) printf("and ");}
        if (MI_algo&MI_ALGO_2) printf("2");
        printf("\n");
/*        if (MI_algo&MASK_CONSERVATIVE)  printf("   -> conservative use of masks (less statistics)\n");
        else                            printf("   -> optimized use of masks (for more statistics)\n");  */
        if (MI_algo&COUNTING_ANN)       printf("   -> (new) counting with ANN\n");
        else                            printf("   -> (legacy) counting optimized\n");
        if (MI_algo&COUNTING_TEST)      printf("   -> using unified counting function\n");
    }
//    printf("variable = %d\n", MI_algo);
//    printf("variable & MI_ALGO_1 = %d\n", (MI_algo&MI_ALGO_1));
    return;
}


int is_equal(double x, double y)
{   const double epsilon=1e-7; // arbitrary
    return(fabs(x - y) <= (epsilon*fabs(x)));
}

int is_zero(double x)
{   const double epsilon=1e-15; // arbitrary
//    const double epsilon=2*DBL_MIN; // DBL_MIN is the min double value on the machine
    return(fabs(x) <= epsilon);
}

void get_last_stds(double *std, double *std2)
{   *std  = last_std;
    *std2 = last_std2;
}

void set_last_stds(double std, double std2)
{   last_std  = std;
    last_std2 = std2;
}


/************************************************************************/
/* to save an array of doubles in a binary file				            */
/************************************************************************/
size_t save_dataset_d(char *filename, double *x1, size_t N)
{	FILE 	*data_file;
	size_t	err;
	
	data_file=fopen(filename, "wb");
	err=fwrite(x1,sizeof(double),(size_t)N,data_file);
/*	if (verbosity>1) printf("save_data_set: file %s save with %d/%d floats.\n",filename,err,N); */
	fclose(data_file);

	return err;
}

/************************************************************************/
/* to save an array of ints in a binary file				            */
/************************************************************************/
size_t save_dataset_i(char *filename, size_t *x1, size_t N)
{	FILE 	*data_file;
	size_t	err;
	
	data_file=fopen(filename, "wb");
	err=fwrite(x1,sizeof(size_t),(size_t)N,data_file);
/*	if (verbosity>1) printf("save_data_set: file %s save with %d/%d floats.\n",filename,err,N); */
	fclose(data_file);

	return err;
}

