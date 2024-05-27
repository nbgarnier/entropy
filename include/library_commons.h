/*
 *  library_commons.h
 *  
 *
 *  Created by Nicolas Garnier on 2020/03/05.
 *  Copyright 2012-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 *  
 */
#ifndef _LIBRARY_COMMONS_H
#define _LIBRARY_COMMONS_H
/**********************************************************************************************************/
// type of algorithms:
#define COUNTING_NG             0x0100  // counting with NG legacy algorithm
                                        // 2020-02-27: changed (0 -> 1)
#define COUNTING_ANN            0x0200  // counting with ANN library (experimental)
                                        // 2020-02-27: changed (1 -> 2)
#define COUNTING_TEST           0x1000  // 2021-12-10: for testing

#define MI_ALGO_1               1       // Kraskov et al algorithm 1 for MI (et al)
#define MI_ALGO_2               2       // Kraskov et al algorithm 2 for MI (t al)

// methods to compute entropy rate:
#define ENTROPY_RATE_FRACTION   0       // H^(m)/m
#define ENTROPY_RATE_DIFFERENCE 1       // H^(m+1)-H^(m)
#define ENTROPY_RATE_MI         2       // H^(1)-MI

// single thread / multi-threads management:
#define SINGLE_TH               1       // when not using pthread, use a single thread (for various mallocs) 


/**********************************************************************************************************/
#ifdef NAN
    #define my_NAN NAN
#else
    #define my_NAN 0.0
#endif


/********************************************************************************************************************/
// choice of the Kraskov et al. algorithm for MIs:
extern int MI_algo;             // defined in entropy_ann.c with values from library_commons.h

/********************************************************************************************************************/
// global variables to count errors, when using wrapper functions or "direct" functions
extern int nb_errors_local;     // defined in entropy_ann.c (before 2020-02-26, old name was "global_nb_errors")
extern int nb_errors;           // errors encountered in a "wrapper" function (usually returned)
extern int nb_errors_total;     // unused as of 2020-02-26

// global variables to count effective number of points used in functions
extern int last_npts_eff;       // effective nb of points used in the last "direct" function (wrapper)
extern int last_npts_eff_local; // effective nb of points used in the last "engine" function

// global variables to keep track of the standard deviation of the last estimate (in a wrapper function)
extern double last_std;         // main variable
extern double last_std2;        // additional variable (for functions returning a second value)

// global variables to record the std (and its std) of the pre-processed data
// (added 2022-03-11, to keep track of basic stats of the input data)
// (useful for increments)
extern double data_std;         // std of the data (e.g., increments, which can be intermediate)
extern double data_std_std;     // std of the std of the data


/********************************************************************************************************************/
struct dimension_parameters {
    int nx;
    int ny;
    int npts;
    int mx;
    int my;
    int mz;
};
typedef struct dimension_parameters dim_param;

struct embedding_parameters {
    int mx;
    int my;
    int mz;
    int stride_x;
    int stride_y;
    int stride_z;
    int lag;
};
typedef struct embedding_parameters embed_param; 

/********************************************************************************************************************/
void ANN_choose_algorithm (int algo);           // to choose the Kraskov et al. algorithm
                                                // as well as to choose the counting algorithm

void get_last_stds(double *std, double *std2);  // to get the value of last_std 
void set_last_stds(double std, double std2);    // to force the value of last_std

int is_equal(double x, double y);               // to test if two doubles are equal, up to some precision
int is_zero (double x);                         // to test if a double is zero, up to some precision

#endif

