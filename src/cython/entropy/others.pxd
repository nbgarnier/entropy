# file others.pxd
#
# this is a copy of functions/definitions that we want to see (and hence use) in Python
#
# 2019-01-24: cleaned up version
# 2021-01-19: removed "bins" functions
# 2021-12-19: forked from ann.pxd



cdef extern from "entropy_ann_mask.h":
	int compute_Renyi_ann_mask  (double *x, char *mask, int npts, 
				int m, int p, int stride, double q, 
				int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S);

cdef extern from "entropy_ann_Renyi.h":
	int compute_Renyi_ann(double *x, int npts, int m, int p, int tau, double q, 
				int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S);


cdef extern from "entropy_others.h":
	double compute_ApEn_old     (double* data, int m, double r, int npts);
	double compute_SampEn_old   (double* data, int m, double r, int npts);
	int compute_complexity		(double* data, int npts, int m, int stride, double r, int kernel_type, double *ApEn, double *SampEn);
	int compute_complexity_mask	(double* data, char *mask, int npts, int m, int stride, 
                            		int tau_Theiler, int N_eff, int N_realizations, 
                            		double r, int kernel_type, double *ApEn, double *SampEn);
                            
