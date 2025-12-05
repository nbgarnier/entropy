# file computes.pxd
#
# this is a copy of functions/definitions that we want to see (and hence use) in Python
#
# 2019-01-24: cleaned up version
# 2021-01-19: removed "bins" functions
# 2022-01-13: re-organisation



cdef extern from "entropy_ann.h":
	int compute_entropy_ann            (double *x, int nx, int m, int p, 
								int tau, int tau_Theiler, int N_eff, int N_realizations, int k, double *S)
	int compute_relative_entropy_ann   (double *x, int nx, double *y, int ny, int mx, int my, int px, int py, 
								int tau, int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *H)
	int compute_mutual_information_ann(double *x, double *y, int nx, int mx, int my, int px, int py,
								int tau, int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2)
	int compute_partial_MI_ann         (double *x, double *y, double *z, int nx, int *dim, 
	                            int tau, int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2)

cdef extern from "entropy_ann_combinations.h": 
	int compute_entropy_increments_ann(double *x, int nx, 
				int n, int p, int stride, int tau_Theiler, int N_eff, int N_realizations, int k, int incr_type, double *S)
		
	int compute_entropy_rate_ann_old(double *x, int nx, int m, int p, int stride, 
				int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *H)
	int compute_entropy_rate_ann(double *x, int nx, int m, int p, int stride, 
				int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *H)				

	int compute_regularity_index_ann(double *x, int npts, int mx, int px, int stride, 
				int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2)
	
	int compute_transfer_entropy_ann(double *x, double *y, int nx, int mx, int my, int px, int py, int stride, int lag, 
				int tau_Theiler, int N_eff, int N_realizations, int k, double *T1, double *T2, int do_sub_Gaussian)

	int compute_directed_information_ann(double *x, double *y, int nx, int mx, int my, int N, int stride, 
				int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2)

cdef extern from "entropy_Gaussian.h":
	int compute_entropy_Gaussian(double *x, int nx, int m, int p, 
								int tau, int tau_Theiler, int N_eff, int N_realizations, double *S)
	int compute_relative_entropy_Gaussian   (double *x, int nx, double *y, int ny, int mx, int my, int px, int py, 
								int tau, int tau_Theiler, int N_eff, int N_realizations, int method, double *H)
	int compute_mutual_information_Gaussian(double *x, double *y, int nx, int mx, int my, int px, int py,
								int tau, int tau_Theiler, int N_eff, int N_realizations, double *I1)
	int compute_partial_MI_Gaussian(double *x, double *y, double *z, int nx, int *dim, 
	                            int tau, int tau_Theiler, int N_eff, int N_realizations, double *I1)

cdef extern from "entropy_Gaussian_combinations.h": 
	int compute_entropy_increments_Gaussian(double *x, int nx, 
				int n, int p, int stride, int tau_Theiler, int N_eff, int N_realizations, int incr_type, double *S)
		
	int compute_entropy_rate_Gaussian(double *x, int nx, int m, int p, int stride, 
				int tau_Theiler, int N_eff, int N_realizations, int method, double *H)

	int compute_regularity_index_Gaussian(double *x, int npts, int mx, int px, int stride, 
				int tau_Theiler, int N_eff, int N_realizations, double *I1)
	
	int compute_transfer_entropy_Gaussian(double *x, double *y, int nx, int mx, int my, int px, int py, int stride, int lag, 
				int tau_Theiler, int N_eff, int N_realizations, double *T1)

	int compute_directed_information_Gaussian(double *x, double *y, int nx, int mx, int my, int N, int stride, 
				int tau_Theiler, int N_eff, int N_realizations, double *I1)

cdef extern from "entropy_ann_2d.h":
	int compute_entropy_ann_2d(double *x, int nx, int ny, int d, int p, int stride_x, int stride_y, 
				int Theiler_x, int Theiler_y, int N_eff, int N_realizations, int k, int method, double *S)

cdef extern from "entropy_ann_mask.h":
	int compute_entropy_ann_mask(double *x, char *mask, int npts, int m, int p, int stride, 
				int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S);

	int compute_Renyi_ann_mask  (double *x, char *mask, int npts, int m, int p, int stride, double q, 
				int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S);

	int compute_entropy_rate_ann_mask(double *x, char *mask, int npts, int m, int p, int stride, 
				int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S)

	int compute_mutual_information_ann_mask(double *x, double *y, char *mask, int npts, int mx, int my, int px, int py, int stride, 
				int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2);

	int compute_transfer_entropy_ann_mask(double *x, double *y, char *mask, int npts, int mx, int my, int px, int py, int stride, int lag, 
				int tau_Theiler, int N_eff, int N_realizations, int k, double *T1, double *T2);

	int compute_partial_TE_ann_mask     (double *x, double *y, double *z, char *mask, int nx, int *dim, int stride, int lag, 
				int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2);

	int compute_partial_MI_ann_mask     (double *x, double *y, double *z, char *mask, int nx, int *dim, int stride, 
				int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2);

	int compute_directed_information_ann_mask(double *x, double *y, char *mask, int npts, int mx, int my, int N, int stride, 
				int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2);


