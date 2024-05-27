/*
 *  math-tools.h
 *  
 *
 *  Created by Nicolas Garnier on 13/08/10.
 *  Copyright 2010-2013 ENS-Lyon CNRS. All rights reserved.
 *
 */

#include <math.h>


/* QuickSort algorithm											*/
void QuickSort_int   (int    *xd, int *yd, int l, int r);
void QuickSort_float (float  *xd, int *yd, int l, int r); /* float version (not used, dangerous !!!)*/
void QuickSort_double(double *xd, int *yd, int l, int r); /* double version */
/* a function to "unsort" the data, and recover initial pointer	:	*/
int unsort_d_i       (double *x, int *ind, int nx);

/* to check that a dataset x has no identical points : returns 0 if it is OK, or the nb of identical pairs */
int check_continuity   (double *x, int nx);
int check_continuity_nd(double *x, int nx, int nd);

/* some norms : */
double	mn(double x1, double y1, double x2, double y2);
// float norm_3d_0(float x1, float y1, float z1, float x2, float y2, float z2);
// float norm_3d_1(float x1, float y1, float z1, float x2, float y2, float z2);
// float norm_3d_2(float x1, float y1, float z1, float x2, float y2, float z2);
// float norm_2d_0(float x1, float y1, float x2, float y2);
// float norm_2d_1(float x1, float y1, float x2, float y2);
// float norm_2d_2(float x1, float y1, float x2, float y2);

double find_min_double(double *x, int nb_pts);
double find_max_double(double *x, int nb_pts);
int    find_min_int   (int    *x, int nx);  /* find the min value in an integer pointer */
int    find_max_int   (int    *x, int nx);  /* find the max value in an integer pointer */
double find_mean      (double *x, int nb_pts);
double find_sigma     (double *x, int nb_pts, double mean);

#define my_min find_min_double
#define my_max find_max_double

/* computes a power of 2, returns 2^p */
int pow2(int p);

int filter_FIR_LP(double *x, int N_pts, int m, int tau, double fr, double *out, int N_pts_new); // LP filter

double determinant(double *M, int n);   // determinant of square matrix
