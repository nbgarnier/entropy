/************************************************************************/
/* entropy_ann_threads_RE.h                                             */
/*								        */
/* 2021-12-09 	                                                        */
/************************************************************************/
/* Nicolas Garnier	nicolas.garnier@ens-lyon.fr		        */
/************************************************************************/


/************************************************************************/
/* Procedures :							        */
/************************************************************************/
double compute_cross_entropy_2xnd_ann_threads(   double *x, int nx, double *y, int ny, int n, int k, int nb_cores);
double compute_relative_entropy_2xnd_ann_threads(double *x, int nx, double *y, int ny, int n, int k, int nb_cores);
