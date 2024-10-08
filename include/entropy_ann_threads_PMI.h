/************************************************************************/
/* entropy_ann_threads_MI.h			                                    */
/*								                                        */
/* 02/12/2021 	                                                        */
/************************************************************************/
/* Nicolas Garnier	nicolas.garnier@ens-lyon.fr			                */
/************************************************************************/



/************************************************************************/
/* Procedures :								                            */
/************************************************************************/
int compute_partial_MI_direct_ann_threads(double *x, int npts, 
            int mx, int my, int mz, int k, double *I1, double *I2, int nb_cores);
