#cython: language_level=3
# Cython wrappers for C functions
#
# 2018-04-10: enhanced memory mapping from C to Python
#       note: only some basic functions have been edited, changes have to be propagated
# 2019-01-24: enhanced tests, interface and documentation + cleaning
# 2020-02-23: now using memoryviews for all functions
#             http://docs.cython.org/en/latest/src/userguide/memoryviews.html#c-and-fortran-contiguous-memoryviews
# 2020-02-23: now using 1-d nd-array for mask (instead of 2-d nd-array)
# 2020-02-26: access to standard deviation of estimators (when stride>1), as well as nb_errors
# 2021-01-20: removed "bins" estimators
# 2021-12-02: multi-threaded entropy algorithm

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_entropy_Renyi(double[:, ::1] x, double q, int inc_type=0, int n_embed=1, int stride=1, 
                    int Theiler=0, int N_eff=0, N_real=0,
                    int k=commons.k_default, char[::1] mask=PNP.zeros(shape=(1),dtype='i1')):
     """
     computes Renyi entropy :math:`H_q` of order :math:`q` of a signal :math:`x` (possibly multi-dimensional) or of its increments, using nearest neighbors search with ANN library.
     (time-)embedding is performed on the fly.
     
     .. math::
          H_q = \\frac{1}{1-q} \\ln \\int p(x^{(m,\\tau)})^q {\\rm d}^m x^{(m,\\tau)}

     (time-)embedding of :math:`x` into :math:`x^{(m, \\tau)}` (see equation :eq:`embedding`) is performed on the fly, see :any:`compute_entropy`          

     :param x: signal (NumPy array with ndim=2, time as second dimension)
     :param q: order of the Renyi entropy (should not be 1)
     :param inc_type: which pre-processing to operate, possible values are:
                0 for entropy of the signal x itself,
                1 for entropy of the increments of the signal x,
                2 for entropy of the averaged increments of x.   
     :param n_embed: embedding dimension :math:`m` (default=1)
     :param stride: stride (or :math`\\tau`) for embedding (default=1) 
     :param Theiler: Theiler scale (should be >= stride, but lower values are tolerated). If Theiler<0, then automatic Theiler is applied as described in function :any:`set_Theiler`.        
     :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
     :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
     :param k: number of neighbors to consider or -1 to force a non-ANN computation using covariance only, assuming Gaussian statistics.
     :param mask: mask to use (NumPy array of dtype=char). If a mask is provided, only values given by the mask will be used. (default=no mask)             
     :returns: the entropy estimate for the increments 
     
     see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.
     """
     
     if (q==1.0): raise ValueError("you want Renyi entropy of order 1, please use Shannon entropy instead")
 
     cdef double S=0.
     cdef int npts=x.shape[1], m=x.shape[0]
     cdef int npts_mask=mask.size
     
     if (npts<m): raise ValueError("please transpose x")
    
     if (Theiler==0): Theiler=commons.samp_default.Theiler
     if (N_eff==0):   N_eff =commons.samp_default.N_eff
     if (N_real==0):  N_real=commons.samp_default.N_real
     
     if (npts_mask>1): # then this is a real mask, not just the default value
        if (npts_mask!=npts):
            raise ValueError("mask does not have the same number of points in time as the data")
        others.compute_Renyi_ann_mask(&x[0,0], &mask[0], npts, m, n_embed, stride, q, Theiler, N_eff, N_real, k, inc_type, &S)
     else:
        others.compute_Renyi_ann     (&x[0,0],           npts, m, n_embed, stride, q, Theiler, N_eff, N_real, k, inc_type, &S)

     return S



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_complexities_old(double[:, ::1] x, int n_embed=1, int stride=1, double r=0.2):
     """
     computes ApEn and SampEn complexities (kernel estimates).
     OLD VERSION - NO ENHANCED SAMPLINGS
     
     :param x: signal (NumPy array with ndim=2, time as second dimension)
     :param n_embed: embedding dimension (default=1)
     :param stride: stride for embedding (default=1) 
     :param r: radius (default=0.2)
     :returns: two nd-arrays of size (n_embed+1). The first array contains ApEn and the second SampEn, each estimate being a function of the embedding dimension, up to the provided value n_embed.
     
     Note that enhanced samplings and/or masking are not available for this function. 
     contact nicolas.b.garnier (@) ens-lyon .fr if interested.
     """
     
     cdef double S=0
     cdef int npts=x.shape[1], m=x.shape[0], ratou
#     cdef int npts_mask=mask.size
     
     if (n_embed<0): raise ValueError("n_embed should be positive")
     if (npts<m):    raise ValueError("please transpose x")
     if (m>1):       raise ValueError("for now, x can only be with dim[0]=1")

     cdef CNP.ndarray[dtype=double, ndim=1] ApEn   = PNP.zeros(n_embed+1, dtype='float')
     cdef CNP.ndarray[dtype=double, ndim=1] SampEn = PNP.zeros(n_embed+1, dtype='float')
#     S = others.compute_ApEn_old  (&x[0,0], n_embed, r, npts)
#     print("ApEn   = %f" %S)
#     S = others.compute_SampEn_old(&x[0,0], n_embed, r, npts)
#     print("SampEn = %f" %S)
     ratou = others.compute_complexity(&x[0,0], npts, n_embed, stride, r, 0, &ApEn[0], &SampEn[0])
#     print("ApEn   = ", ApEn)
#     print("SampEn = ", SampEn)
     return ApEn, SampEn



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_complexities(double[:, ::1] x, int n_embed=1, int stride=1, double r=0.2,
                            int Theiler=0, int N_eff=0, N_real=0,
                            char[::1] mask=PNP.zeros(shape=(1),dtype='i1')):
     """
     computes ApEn and SampEn complexities (kernel estimates).
 
     :param x: signal (NumPy array with ndim=2, time as second dimension)
     :param n_embed: embedding dimension (default=1)
     :param stride: stride for embedding (default=1) 
     :param r: radius (default=0.2)
     :param Theiler: Theiler scale (should be >= stride, but lower values are tolerated). If Theiler<0, then automatic Theiler is applied as described in function :any:`set_Theiler`.        
     :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
     :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
     :param mask: mask to use (NumPy array of dtype=char). If a mask is provided, only values given by the mask will be used. (default=no mask)
     :returns: two nd-arrays of size (n_embed+1). The first array contains ApEn and the second SampEn, each estimate being a function of the embedding dimension, up to the provided value n_embed.
     
     see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.
     """
     
     cdef double S=0
     cdef int npts=x.shape[1], m=x.shape[0]
     cdef int npts_mask=mask.size

     if (Theiler==0): Theiler=commons.samp_default.Theiler
     if (N_eff==0):   N_eff =commons.samp_default.N_eff
     if (N_real==0):  N_real=commons.samp_default.N_real
     
     if (n_embed<0): raise ValueError("n_embed should be positive (or 0)")
     if (npts<m):    raise ValueError("please transpose x")
    
     cdef CNP.ndarray[dtype=double, ndim=1] ApEn   = PNP.zeros(n_embed+1, dtype='float')
     cdef CNP.ndarray[dtype=double, ndim=1] SampEn = PNP.zeros(n_embed+1, dtype='float')
     
     if (npts_mask>1): # then this is a real mask, not just the default value
          if (npts_mask!=npts): raise ValueError("mask does not have the same number of points in time as the data")
     else:
#       mask = PNP.ones(shape=(1,npts), dtype='i1') # arbitrary convention #1 (deprecated)
          mask = PNP.ones(npts, dtype='i1')           # simpler arbitrary convention 

     ratou = others.compute_complexity_mask(&x[0,0], &mask[0], npts, n_embed, stride, Theiler, N_eff, N_real, r, 0, &ApEn[0], &SampEn[0])
#     print("ApEn   = ", ApEn)
#     print("SampEn = ", SampEn)
     return ApEn, SampEn

