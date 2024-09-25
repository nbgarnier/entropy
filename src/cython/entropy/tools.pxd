# file tools.pxd
#
# this is a copy of functions/definitions that we want to see (and hence use) in Python
#
# 2024-09-24: new for tools.pyx (replaces tools.py)

# import definitions from C : 
cdef extern from "entropy_ann.h": 
	const int k_default;
	

cdef extern from "math_tools.h":
	int filter_FIR_LP(double *x, int N_pts, int m, int tau, double fr, double *out, int N_pts_new)	


cdef extern from "samplings.h":
	struct sampling_parameters:
		int Theiler        
		int Theiler_max  
		int N_eff
		int N_eff_max;      
		int N_real;
		int N_real_max;    
		int type;         
	ctypedef sampling_parameters samp_param
	samp_param samp_default
	samp_param last_samp
	struct sampling_parameters_extra_2d:
		int last_Theiler_x
		int last_Theiler_y
		int type;      
	ctypedef sampling_parameters_extra_2d samp_param_2d
	samp_param_2d samp_2d
	int Theiler_nb_pts_new  (int npts, int stride, int n_embed_max)
	void time_embed(double *data, double *output, int npts, int npts_new, int nb_dim, int n_embed, int n_embed_max, int stride, int i_start, int n_window)
	void crop_array(double *data, double *output, int npts, int nb_dim, int npts_new, int i_window)

cdef extern from "surrogates.h":
	void surrogate_uFt		(double *x, int npts, int mx)
	void surrogate_wFt		(double *x, int npts, int mx)
	void surrogate_aaFt		(double *x, int npts, int mx)
	void surrogate_improved	(double *x, int npts, int mx, int N_steps)
	void Gaussianize		(double *x, int npts, int mx)
	void shuffle_data		(double *x, int npts, int mx)
	double *create_surrogate(double *x, int npts, int mx)


cdef extern from "mask.h":
	int NaN_mask	(double *x, int npts, int m, char *mask);
	int finite_mask (double *x, int npts, int m, char *mask);	
