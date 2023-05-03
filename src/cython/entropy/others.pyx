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
def compute_entropy_Renyi(double[:, ::1] x, double q, int n_embed=1, int stride=1, 
                    int Theiler=0, int N_eff=0, N_real=0,
                    int k=commons.k_default, int method=0, char[::1] mask=PNP.zeros(shape=(1),dtype='i1')):
     ''' H = compute_entropy_Renyi(x, q, n_embed=1, stride=1, method=0, ..., mask)
    
     computes Renyi entropy of order q of a vector x (possibly multi-dimensional)
     using nearest neighbors search with ANN library.
     embedding is performed on the fly.
    
     x        : signal (NumPy array with ndim=2, time as second dimension)
     q        : order of the Renyi entropy
     n_embed  : embedding dimension (default=1)
     stride   : stride (Theiler correction will be used accordingly, even if n_embed=1) (default=1)
     method   : which entropy to compute, possible values are:
                0 for regular entropy
                1 for entropy of the increments
                2 for entropy of the averaged increments
     Theiler  : Theiler correction (should be >= stride, but lower values are tolerated) 
             !!! if Theiler<0, then automatic Theiler is applied as follows:
               -1 for Theiler=tau, and uniform sampling (thus localized in the dataset) (legacy)
               -2 for Theiler=max, and uniform sampling (thus covering the all dataset)
               -3 for Theiler=tau, and random sampling 
               -4 for Theiler=max, and random sampling (default)         
     N_eff    : nb of points to consider in the statistics (default=4096)
               -1 for legacy behavior (largest possible value)
     N_real   : nb of realizations to consider (default=10)
               -1 for legacy behavior (N_real=stride)
     k        : number of neighbors to consider (typically 7 or 10) (default=5)
     mask     : mask to use (NumPy array of dtype=char) (default=no mask)
                if a mask is provided, only values given by the mask will be used
     '''
     
     if (q==1.0):
         raise ValueError("you want Renyi entropy of order 1, please use Shannon entropy instead")
 
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
        others.compute_Renyi_ann_mask(&x[0,0], &mask[0], npts, m, n_embed, stride, q, Theiler, N_eff, N_real, k, method, &S)
     else:
        others.compute_Renyi_ann     (&x[0,0],           npts, m, n_embed, stride, q, Theiler, N_eff, N_real, k, method, &S)

     return S



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_complexities(double[:, ::1] x, int n_embed=1, int stride=1, double r=1.0):
     ''' ApEn, SampEn = compute_complexities(x, [n_embed, stride, r])
     
     computes complexities.
     
     x        : signal (NumPy array with ndim=2, but unidimensional)
     n_embed  : embedding dimension (default=1)
     stride   : stride (no Theiler correction for ApEn or SampEn) (default=1)
     r        : radius (default=0.2)
#     mask     : mask to use (NumPy array of dtype=char) (default=no mask)
#                if a mask is provided, only values given by the mask will be used
     
     two arrays of size (n_embed+1) are returned, the first for ApEn, the second for SampEn 
     '''
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
     ratou = others.compute_complexity    (&x[0,0], npts, n_embed, stride, r, 0, &ApEn[0], &SampEn[0])
#     print("ApEn   = ", ApEn)
#     print("SampEn = ", SampEn)
     return ApEn, SampEn

