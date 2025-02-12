/*
 *  main.c
 *  
 *
 *  Created by Nicolas Garnier on 26/07/10.
 *  Copyright 2010-2024 ENS-Lyon CNRS. All rights reserved.
 *
 */

// #include <unistd.h>  // to probe the nb of processors
#include <pthread.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
//#include "inout.h"
#include "math_tools.h"
#include "library_commons.h"
#include "verbosity.h"
#include "entropy_ann.h"
#include "entropy_ann_combinations.h"
#include "entropy_ann_threads.h"
#include "samplings.h" 	// 2022-12-07 for tests
#include "surrogates.h" // 2023-03-04 for tests
//#include "integration-cml.h"
//#include "stat.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define verbosity 2

#define IS_GAUSSIAN 0x0001
#define IS_FLAT     0x0010

#define TIMING
#define DO_ENTROPY 0
#define DO_MI      01
#define DO_PMI     0
#define DO_THREADS 0
#define DO_PERM    1

#ifdef TIMING
    #include "timings.h"
#else 
    #define tic()
    #define toc(x) 0.0
#endif
    

/* set the initial conditions */
void initialize_data(double *x, int n, int distribution_type)
{	register int i;
	const gsl_rng_type * T;
    static	gsl_rng * r;
    static int has_run=0;
    
    if (has_run==0) 
    {   has_run=1;
        /* create a generator chosen by the environment variable GSL_RNG_TYPE */
	    gsl_rng_env_setup();
	    gsl_rng_default_seed = time(NULL);
	
	    T = gsl_rng_knuthran; //or: gsl_rng_default;
	    if (r==NULL)	r = gsl_rng_alloc (T);
    }
    
	for (i=0; i<n; i++)
	{	if (distribution_type & IS_FLAT) 
		{	x[i] = gsl_ran_flat     (r, 0.0, 1.0); }
		if (distribution_type & IS_GAUSSIAN)
		{	x[i] = gsl_ran_gaussian (r, 1.0);	}
	}
//	gsl_rng_free(r);
	return;
}

/************************************************************************/
size_t load_dataset_d(char *filename, double **x1)
/* to load an array of float as a variable				*/
/* complex number are stored as 2 consecutive real numbers for real and */
/*	imaginary part. So a n-dimensionnal vector of complex contains  */
/*	2*n real numbers.						*/
/************************************************************************/
{	FILE 	*data_file;
	int	N;
	size_t i;
	double 	*x;
	
	data_file=fopen(filename, "rb");
	if (data_file==NULL) printf("load_data_set: error opening file %s\n",filename);
	
	x =(double*)calloc(1,sizeof(float)); 
	N=-1;
	while (!feof(data_file)) 
	{ 	i=fread(x,sizeof(double),1,data_file); 
		N++; 
	}
	free(x);
	rewind(data_file);
	*x1=(double*)calloc((size_t)N,sizeof(double));
	i=fread(*x1,sizeof(double),(size_t)N,data_file);
	
	fclose(data_file);
	return i;
}




int compare(double *x, double *y, int npts)
{   register int i;
    int     nb_pbs=0;
    
    for (i=0; i<npts; i++) 
        if (x[i]!=y[i]) nb_errors++;
        else if (!is_zero(x[i]-y[i])) nb_pbs++;
        
    return(nb_pbs);
}

    
int main(void)
{	int nx, pb=0;
//	double *x=NULL;
	double H1=0, H2=0;
	int PMI_dims[6]={1,1,1,2,1,1};
	int px=1, py=1; // embedding dimensions (for MI or PMI)
	register int k;
	double temps; // for timings
//	char filename[128] = "donnees_erreur_emb7_lag6.dat"; //"donnees_erreur_emb2_lag1.dat";
	int i, stride=777;
	double *x=NULL, *y=NULL, *z=NULL, *xc=NULL, *yc=NULL, *zc=NULL;
	int Theiler=4, N_eff=4000, N_real=10;
	
	lib_verbosity=1;
	
	nx = 100000;	// nb de points
	k  = 5;		// nb de voisins

	x  = (double*)calloc(nx, sizeof(double));
	y  = (double*)calloc(nx, sizeof(double));
	z  = (double*)calloc(nx, sizeof(double));
	initialize_data(x, nx, IS_GAUSSIAN); // tirage Gaussien, 
	initialize_data(y, nx, IS_GAUSSIAN); // tirage Gaussien, 
	initialize_data(z, nx, IS_GAUSSIAN); // tirage Gaussien, 
	// add some correlations:
//	for (i=0; i<nx; i++) y[i] += x[i]/2;
	for (i=0; i<nx; i++) 
	    if (x[i]==y[i]) printf("i= %d ; identical value between x and y%f\n", i, x[i]);
	// for debug:
/*	for (i=0; i<nx; i++) 
	{   x[i] = i;
	    y[i] /= 10;
	    y[i] += x[i]/1;
	}
*/
    if ((pb=check_continuity_nd(x, nx, 1))>0) printf("x has %d identical pts\n", pb);
    if ((pb=check_continuity_nd(y, nx, 1))>0) printf("y has %d identical pts\n", pb);
//	save_dataset_d("test2-x-100pts.dat", x, nx);
//	save_dataset_d("test2-y-100pts.dat", y, nx);

//    printf("load x: %zu pts  ",load_dataset_d("test2-x-100pts.dat", &x));
//    printf("load x: %zu pts\n",load_dataset_d("test2-y-100pts.dat", &y));
    
    // copy x for backup:
    xc=(double*)calloc(nx, sizeof(double));
    yc=(double*)calloc(nx, sizeof(double));
    zc=(double*)calloc(nx, sizeof(double));
    memcpy(xc, x, nx*sizeof(double));
    memcpy(yc, y, nx*sizeof(double));
    memcpy(zc, z, nx*sizeof(double));
    
 	printf("signals with %d points, min(x)=%f, max(x)=%f\n", nx, my_min(x,nx), my_max(x,nx) );
 	printf("\n");
 	
 	printf("%d\n", NCORES);
 	get_cores_number(0); // we print some informations
 	get_multithreading_state(1); // we print some informations
 	printf("\n");
 	
//	printf("for %d points, optimal cores number is %d\n", nx, adapt_cores_number(nx));
//	printf("for 4096 points, optimal cores number is %d\n", adapt_cores_number(4096));
//	printf("code will use %d threads\n", get_cores_number(GET_CORES_SELECTED));
//	set_cores_number(adapt_cores_number(nx));
//	get_cores_number(0); // we print some informations
//	set_multithreading_state(1);
// 	get_multithreading_state(1); // we print some informations

	printf("code will use %d threads\n", get_cores_number(GET_CORES_SELECTED));
	
 	if (DO_ENTROPY)
    {   temps=0; tic();
	    pb = compute_entropy_ann(x, nx, 1, px, stride, Theiler, N_eff, N_real, k, &H1);
	    toc(&temps);
	    if (pb!=0) printf("\tregular entropy returned %d\n", pb);
        printf("entropy = %f\t elapsed time = %.3f s (auto-adapted multi-threading)\n\n", H1, temps);
        
        temps=0; tic();
	    pb = compute_entropy_increments_ann(x, nx, 1, 1, stride, -1, -1, -1, k, 0, &H1);
	    toc(&temps);
	    if (pb!=0) printf("\tincrements entropy returned %d\n", pb);
        printf("entropy (increments) = %f\t elapsed time = %.3f s (auto-adapted multi-threading)\n", H1, temps);
        printf("\n");
	}
	
    if (DO_MI)
	{   ANN_choose_algorithm((1|2));
    
/*    	// added 2023-03-04:
    	set_multithreading_state(0);
    	printf("code will use %d threads\n", get_cores_number(GET_CORES_SELECTED));
    	surrogate_improved(xc, nx, 1, 10);
*/

// test new samplings, type 1
        samp_default.Theiler=-1; // impose legacy behavior
//        printf_sampling_parameters(samp_default, "samp_default before _N type 1"); // for debug

        temps=0; tic();
	    pb = compute_mutual_information_ann(x,y, nx, 1, 1, px, py, stride, 
	                                samp_default.Theiler, samp_default.N_eff, samp_default.N_real, k, &H1, &H2);
        toc(&temps);
//        printf_sampling_parameters(last_samp, "last_samp after _N type 1"); // for debug

	    if (pb!=0) printf("\tregular MI (NBG) (_N type 1) returned %d\n", pb);
	    printf("MI = %8.5f +/- %f / %8.5f +/- %f\t elapsed time = %.3f s (auto-adapted multi-threading) (NBG) (_N type 1)\n", 
	    			H1, last_std, H2, last_std2, temps);
        if (compare(x, xc, nx)>0) printf("\t!!! signal x modified !!!\n");
        if (compare(y, yc, nx)>0) printf("\t!!! signal y modified !!!\n");
    	printf_sampling_parameters(last_samp, "last_samp"); // for debug
    	printf("\n");
    	
// test new samplings, type 2
        samp_default.Theiler=-2; // impose legacy behavior
//        printf_sampling_parameters(samp_default, "samp_default before _N type 2"); // for debug

        temps=0; tic();
	    pb = compute_mutual_information_ann(x,y, nx, 1, 1, px, py, stride, 
	                                samp_default.Theiler, samp_default.N_eff, samp_default.N_real, k, &H1, &H2);
        toc(&temps);
//        printf_sampling_parameters(last_samp, "last_samp after _N type 2"); // for debug

	    if (pb!=0) printf("\tregular MI (NBG) (_N type 2) returned %d\n", pb);
	    printf("MI = %8.5f +/- %f / %8.5f +/- %f\t elapsed time = %.3f s (auto-adapted multi-threading) (NBG) (_N type 2)\n", 
	    			H1, last_std, H2, last_std2, temps);
        if (compare(x, xc, nx)>0) printf("\t!!! signal x modified !!!\n");
        if (compare(y, yc, nx)>0) printf("\t!!! signal y modified !!!\n");
    	printf_sampling_parameters(last_samp, "last_samp"); // for debug
    	printf("\n");
    	
// test new samplings, type 3  
        samp_default.Theiler=-3;
        samp_default.N_eff = 4096;
//        printf_sampling_parameters(samp_default, "samp_default before _N type 4"); // for debug  
        temps=0; tic();
	    pb = compute_mutual_information_ann(x,y, nx, 1, 1, px, py, stride, 
	                                samp_default.Theiler, samp_default.N_eff, samp_default.N_real, k, &H1, &H2);
        toc(&temps);
//        printf_sampling_parameters(last_samp, "last_samp after _N type 4"); // for debug

	    if (pb!=0) printf("\tregular MI (NBG) (_N, type 3) returned %d\n", pb);
	    printf("MI = %8.5f +/- %f / %8.5f +/- %f\t elapsed time = %.3f s (auto-adapted multi-threading) (NBG) (_N type 3)\n",
	    			H1, last_std, H2, last_std2, temps);
	    if (compare(x, xc, nx)>0) printf("\t!!! signal x modified !!!\n");
        if (compare(y, yc, nx)>0) printf("\t!!! signal y modified !!!\n");
    	printf_sampling_parameters(last_samp, "last_samp"); // for debug
    	printf("\n");

// test new samplings, type 4
		samp_default.Theiler=-4;
        samp_default.N_eff = 4096;
//        printf_sampling_parameters(samp_default, "samp_default before _N type 4"); // for debug  
        temps=0; tic();
	    pb = compute_mutual_information_ann(x,y, nx, 1, 1, px, py, stride, 
	                                samp_default.Theiler, samp_default.N_eff, samp_default.N_real, k, &H1, &H2);
        toc(&temps);
//        printf_sampling_parameters(last_samp, "last_samp after _N type 4"); // for debug

	    if (pb!=0) printf("\tregular MI (NBG) (_N, type 4) returned %d\n", pb);
	    printf("MI = %8.5f +/- %f / %8.5f +/- %f\t elapsed time = %.3f s (auto-adapted multi-threading) (NBG) (_N type 4)\n",
	    			H1, last_std, H2, last_std2, temps);
	    if (compare(x, xc, nx)>0) printf("\t!!! signal x modified !!!\n");
        if (compare(y, yc, nx)>0) printf("\t!!! signal y modified !!!\n");
    	printf_sampling_parameters(last_samp, "last_samp"); // for debug
    	printf("\n");
    	
        ANN_choose_algorithm((1|2)+COUNTING_ANN);
        temps=0; tic();
	    pb = compute_mutual_information_ann(x,y, nx, 1, 1, px, py, stride, Theiler, N_eff, N_real, k, &H1, &H2);
        toc(&temps);
	    if (pb!=0) printf("\tregular MI (ANN) returned %d\n", pb);
	    printf("MI = %8.5f +/- %f / %8.5f +/- %f\t elapsed time = %.3f s (auto-adapted multi-threading) (ANN)\n",
	    			H1, last_std, H2, last_std2, temps);
        if (compare(x, xc, nx)>0) printf("\t!!! signal x modified !!!\n");
        if (compare(y, yc, nx)>0) printf("\t!!! signal y modified !!!\n");
        printf("\n"); 
    }

    if (DO_PMI)
	{   ANN_choose_algorithm((1|2));
    
	    temps=0; tic();
	    pb = compute_partial_MI_ann(x,y,z, nx, PMI_dims, stride, Theiler, N_eff, N_real, k, &H1, &H2);
        toc(&temps);
	    if (pb!=0) printf("\tregular PMI returned %d\n", pb);
	    printf("PMI = %f / %f\t elapsed time = %.3f s (auto-adapted multi-threading) (NBG)\n\n", H1, H2, temps);
        if (compare(x, xc, nx)>0) printf("\t!!! signal x modified !!!\n");
        if (compare(y, yc, nx)>0) printf("\t!!! signal y modified !!!\n");
        if (compare(z, zc, nx)>0) printf("\t!!! signal z modified !!!\n");
    
        ANN_choose_algorithm((1|2)+COUNTING_ANN);
        temps=0; tic();
	    pb = compute_partial_MI_ann(x,y,z, nx, PMI_dims, stride, Theiler, N_eff, N_real, k, &H1, &H2);
        toc(&temps);
	    if (pb!=0) printf("\tregular PMI returned %d\n", pb);
	    printf("PMI = %f / %f\t elapsed time = %.3f s (auto-adapted multi-threading) (ANN)\n\n", H1, H2, temps);
        if (compare(x, xc, nx)>0) printf("\t!!! signal x modified !!!\n");
        if (compare(y, yc, nx)>0) printf("\t!!! signal y modified !!!\n");
        if (compare(z, zc, nx)>0) printf("\t!!! signal z modified !!!\n");
    }
    
    if (DO_THREADS) {
    set_multithreading_state(1); // 1 means fixed nb of threads
	
	if (DO_ENTROPY)
    {	for (i=0; i<=get_cores_number(GET_CORES_AVAILABLE)*0+4; i++)
	    {   set_cores_number((int)exp2(i));
 	        temps=0; tic();
	        pb = compute_entropy_ann(x, nx, 1, px, stride, Theiler, N_eff, N_real, k, &H2);
	        toc(&temps);
	        if (pb!=0) printf("\tthreaded entropy returned %d\n", pb);
	        printf("entropy = %f\t elapsed time = %.3f s with %d core(s)\n", H2, temps, get_cores_number(GET_CORES_SELECTED));
        }
    }
	  
	if (DO_MI)
	{	for (i=0; i<=get_cores_number(GET_CORES_AVAILABLE)*0+4; i++)
	    {   set_cores_number((int)exp2(i));

            ANN_choose_algorithm((1|2));
            temps=0; tic();
	        pb = compute_mutual_information_ann(x,y, nx, 1, 1, px, py, stride, Theiler, N_eff, N_real, k, &H1, &H2);
            toc(&temps);
	        if (pb!=0) printf("\tregular MI (NBG counting) returned %d\n", pb);
	        printf("MI = %f / %f\t elapsed time = %.3f s with %2d core(s) (NBG counting)\n\n", H1, H2, temps, get_cores_number(GET_CORES_SELECTED));
            if (compare(x, xc, nx)>0) printf("\t!!! signal x modified !!!\n");
            if (compare(y, yc, nx)>0) printf("\t!!! signal y modified !!!\n");
        }
        
        for (i=0; i<=get_cores_number(GET_CORES_AVAILABLE)*0+4; i++)
	    {   set_cores_number((int)exp2(i));

            ANN_choose_algorithm((1|2)+COUNTING_ANN);
            temps=0; tic();
	        pb = compute_mutual_information_ann(x,y, nx, 1, 1, px, py, stride, Theiler, N_eff, N_real, k, &H1, &H2);
            toc(&temps);
	        if (pb!=0) printf("\tregular MI (ANN counting) returned %d\n", pb);
	        printf("MI = %f / %f\t elapsed time = %.3f s with %2d core(s) (ANN counting)\n\n", H1, H2, temps, get_cores_number(GET_CORES_SELECTED));
            if (compare(x, xc, nx)>0) printf("\t!!! signal x modified !!!\n");
            if (compare(y, yc, nx)>0) printf("\t!!! signal y modified !!!\n");
        }
    } // DO_MI
    
    if (DO_PMI)
	{	for (i=0; i<=get_cores_number(GET_CORES_AVAILABLE)*0+4; i++)
	    {   set_cores_number((int)exp2(i));
	    
	        ANN_choose_algorithm((1|2));
    	    temps=0; tic();
	        pb = compute_partial_MI_ann(x,y,z, nx, PMI_dims, stride, Theiler, N_eff, N_real, k, &H1, &H2);
            toc(&temps);
	        if (pb!=0) printf("\tthreated PMI returned %d\n", pb);
	        printf("PMI = %f / %f\t elapsed time = %.3f s with %d core(s) (NBG counting)\n\n", H1, H2, temps, get_cores_number(GET_CORES_SELECTED));
   	        if (compare(x, xc, nx)>0) printf("\t!!! signal x modified !!!\n");
            if (compare(y, yc, nx)>0) printf("\t!!! signal y modified !!!\n");
            if (compare(z, zc, nx)>0) printf("\t!!! signal z modified !!!\n");
        }
        
        for (i=0; i<=get_cores_number(GET_CORES_AVAILABLE)*0+4; i++)
	    {   set_cores_number((int)exp2(i));
	    
	        ANN_choose_algorithm((1|2)+COUNTING_ANN);
    	    temps=0; tic();
	        pb = compute_partial_MI_ann(x,y,z, nx, PMI_dims, stride, Theiler, N_eff, N_real, k, &H1, &H2);
            toc(&temps);
	        if (pb!=0) printf("\tthreated PMI returned %d\n", pb);
	        printf("PMI = %f / %f\t elapsed time = %.3f s with %d core(s) (ANN counting)\n\n", H1, H2, temps, get_cores_number(GET_CORES_SELECTED));
   	        if (compare(x, xc, nx)>0) printf("\t!!! signal x modified !!!\n");
            if (compare(y, yc, nx)>0) printf("\t!!! signal y modified !!!\n");
            if (compare(z, zc, nx)>0) printf("\t!!! signal z modified !!!\n");
        }
    } // DO_PMI
    } // end "DO_THREADS"
    
   	free(x); free(y); free(z); free(xc); free(yc); free(zc);
    return(0);
}



