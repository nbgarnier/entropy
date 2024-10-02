# file commons.pxd
#
# this is a copy of functions/definitions that we want to see (and hence use) in Python
#
# 2019-01-24: cleaned up version
# 2021-01-19: removed "bins" functions
# 2021-12-19: forked out of entropy.pxd

# import definitions from C : 
cdef extern from "library_commons.h":
	int		nb_errors, nb_errors_local, nb_errors_total
	int		last_npts_eff, last_npts_eff_local
	double	last_std, last_std2
	double  data_std, data_std_std
	struct dimension_parameters:
		int nx;
		int ny;
		int npts;
		int dim;
	struct embedding_parameters:
		int mx;
		int my;
		int stride_x;
		int stride_y;
		int lag;
	void ANN_set_verbosity    (int level)	
	void ANN_choose_algorithm (int algo)
	void get_last_stds        (double *std, double *std2)
	void set_last_stds        (double std,  double std2)


cdef extern from "verbosity.h":
	int lib_verbosity          
	int lib_warning_level     


cdef extern from "entropy_ann_threads.h":
	int 	USE_PTHREAD
	void 	set_multithreading_state(int do_mp)
	int  	get_cores_number  		(int get_what)
	void 	set_cores_number  		(int n)
	int 	adapt_cores_number		(int npts_eff)


cdef extern from "entropy_ann.h": 
	const int k_default;
	

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
