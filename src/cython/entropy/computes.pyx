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
# 2021-12-19: forked from "entropy_ann.pyx" into this file "computes.pyx"
# 2022-05-19: sampling parameters selection (unfinished)
# 2022-05-23: sampling parameters selection in function calls, and from default set : OK
# 2022-10-11: Gaussian estimates now included
# 2022-11-26: change in library hierarchy
# 2023-11-28: legacy samplings removed from Python library calls

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_entropy( double[:, ::1] x, int n_embed=1, int stride=1, 
                int Theiler=0, int N_eff=0, int N_real=0,
                int k=commons.k_default, char[::1] mask=PNP.zeros(shape=(1),dtype='i1')):
    """     
    compute Shannon entropy :math:`H` of a signal :math:`x` (possibly multi-dimensional) using nearest neighbors search with ANN library.
    
    .. math::
        H = - \\int p(x^{(m,\\tau)}) \\ln p(x^{(m,\\tau)}) {\\rm d}^m x^{(m,\\tau)}

    (time-)embedding of :math:`x` into :math:`x^{(n_embed, stride)}` is performed on the fly:

    .. math::
        x \\rightarrow x^{(m,\\tau)}=(x_t, x_{t-\\tau}, ..., x_{t-(m-1)\\tau})
        :label: embedding
    
    where :math:`m` is given by the parameter n_embed and :math:`\\tau` is given by the parameter stride.   
    
    :param x: signal (NumPy array with ndim=2, time along second dimension)
    :param n_embed: embedding dimension (default=1)
    :param stride: stride (e.g., time scale \\tau) for embedding (default=1) 
    :param Theiler: Theiler scale (should be >= stride, but lower values are tolerated). If Theiler<0, then automatic Theiler is applied as described in function :any:`set_Theiler`.        
    :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
    :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
    :param k: number of neighbors to consider or -1 to force a non-ANN computation using covariance only, assuming Gaussian statistics.
    :param mask: mask to use (NumPy array of dtype=char). If a mask is provided, only values given by the mask will be used. (default=no mask)
                 
    :returns: the entropy estimate
     
    see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.
    """
    cdef double S=0
    cdef int npts=x.shape[1], m=x.shape[0], ratou # 2018-04-13: carefull with ordering of dimensions!
    cdef int npts_mask=mask.size
     
    if (npts<m):     raise ValueError("please transpose x")
    if (Theiler==0): Theiler=commons.samp_default.Theiler
    if (N_eff==0):   N_eff =commons.samp_default.N_eff
    if (N_real==0):  N_real=commons.samp_default.N_real
       
    if (k==-1):
#        print("Gaussian entropy")
        ratou = computes.compute_entropy_Gaussian(&x[0,0], npts, m, n_embed, stride, Theiler, N_eff, N_real, &S)
        return S  
        
    if (npts_mask>1): # then this is a real mask, not just the default value
        if (npts_mask!=npts): raise ValueError("mask and data do not have the same number of points in time")
        ratou = computes.compute_entropy_ann_mask(&x[0,0], &mask[0], npts, m, n_embed, stride, Theiler, N_eff, N_real, k, 0, &S)
    else:  
        ratou = computes.compute_entropy_ann     (&x[0,0], npts, m, n_embed, stride, Theiler, N_eff, N_real, k, &S)
        
    return S



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_entropy_increments( double[:, ::1] x, int inc_type=1, int order=1, int stride=1, 
                int Theiler=0, int N_eff=0, int N_real=0,
                int k=commons.k_default, char[::1] mask=PNP.zeros(shape=(1),dtype='i1')):
     """
     compute entropy of the increments of a signal (possibly multi-dimensional) using nearest neighbors search with ANN library.
     
      .. math::
        H(\\delta_\\tau x) = - \\int p(\\delta_\\tau x) \\ln p(\\delta_\\tau x) {\\rm d} \\delta_\\tau x

     (time-)increments :math:`\\delta_\\tau x` are computed from signal :math:`x` on the fly:
     
     .. math::
        x \\rightarrow \\delta_\\tau x &= x_t - x_{t-\\tau} \\quad {\\rm (regular \, increments \, of \, order \, 1)} \\\\
        &= x_t - \\sum_{k=1}^{\\tau} x_{t-k} \\quad {\\rm (averaged \, increments \, of \, order \, 1}
    
     :param x: signal (NumPy array with ndim=2, time along second dimension)
     :param inc_type: increments type (regular or averaged):

      - 1 for regular increments (of given order)
      - 2 for averaged increments (of order 1 only)

     :param order: order of increments (between 0 and 5) (default=1)
     :param stride: stride for embedding (default=1) 
     :param Theiler: Theiler scale (should be >= stride, but lower values are tolerated). If Theiler<0, then automatic Theiler is applied as described in function :any:`set_Theiler`.        
     :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
     :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
     :param k: number of neighbors to consider or -1 to force a non-ANN computation using covariance only, assuming Gaussian statistics.
     :param mask: mask to use (NumPy array of dtype=char). If a mask is provided, only values given by the mask will be used. (default=no mask)
                 
     :returns: the entropy estimate for the increments 
     
     see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.
     """
     cdef double S=0
     cdef n_embed=order
     cdef int npts=x.shape[1], m=x.shape[0], npts_mask=mask.size, ratou
     
     if (npts<m):     raise ValueError("please transpose x")
     if ( (inc_type!=1) & (inc_type!=2) ): raise ValueError("increments type must be either 1 or 2")
     if (Theiler==0): Theiler=commons.samp_default.Theiler
     if (N_eff==0):   N_eff =commons.samp_default.N_eff
     if (N_real==0):  N_real=commons.samp_default.N_real
     
     if (k==-1):
        ratou = computes.compute_entropy_increments_Gaussian(&x[0,0], npts, m, n_embed, stride, 
                                    Theiler, N_eff, N_real,    inc_type, &S)
        return S  
        
     if (npts_mask>1): # then this is a real mask, not just the default value
        if (npts_mask!=npts): raise ValueError("mask does not have the same number of points in time as the data")
        ratou = computes.compute_entropy_ann_mask           (&x[0,0], &mask[0], npts, m, n_embed, stride, 
                                    Theiler, N_eff, N_real, k, inc_type, &S)
     else:
        ratou = computes.compute_entropy_increments_ann     (&x[0,0], npts, m, n_embed, stride, 
                                    Theiler, N_eff, N_real, k, inc_type, &S)

     return S



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_entropy_rate( double[:, ::1] x, int method=2, int m=1, int stride=1,
            int Theiler=0, int N_eff=0, int N_real=0,
            int k=commons.k_default, char[::1] mask=PNP.zeros(shape=(1),dtype='i1')):
     """
     computes entropy rate of order m of a vector (possibly multi-dimensional) using nearest neighbors search with ANN library.
     (time-)embedding (see equation :eq:`embedding`) is performed on the fly.           

     :param x: signal (NumPy array with ndim=2, time along second dimension)
     :param method: an integer. in {0,1,2} to indicate which method to use (default=2) (see below)
     :param m: embedding dimension (default=1)
     :param stride: stride for embedding (default=1) 
     :param Theiler: Theiler scale (should be >= stride, but lower values are tolerated). If Theiler<0, then automatic Theiler is applied as described in function :any:`set_Theiler`.        
     :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
     :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
     :param k: number of neighbors to consider or -1 to force a non-ANN computation using covariance only, assuming Gaussian statistics.
     :param mask: mask to use (NumPy array of dtype=char). If a mask is provided, only values given by the mask will be used. (default=no mask)
                 
     :returns: the entropy estimate
     
     The parameter method influences the computation as follows:       
      - 0 for :math:`H^{(m)}/m`
      - 1 for :math:`H^{(m+1)}-H^{(m)}`  
      - 2 for :math:`H^{(1)}-MI(x,x^{(m)})`    (default=2)
     
     see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.
     """     
     cdef double S=0
     cdef int npts=x.shape[1], nx=x.shape[0], ratou 
     cdef int npts_mask=mask.size
     
     if (npts<nx):    raise ValueError("please transpose x")
     if (Theiler==0): Theiler=commons.samp_default.Theiler
     if (N_eff==0):   N_eff =commons.samp_default.N_eff
     if (N_real==0):  N_real=commons.samp_default.N_real              
     
     if (k==-1):
        ratou = computes.compute_entropy_rate_Gaussian(&x[0,0],           npts, nx, m, stride, 
                                    Theiler, N_eff, N_real,    method, &S)
        return S

     if (npts_mask>1): # then this is a real mask, not just the default value
        if (npts_mask!=npts):
            raise ValueError("mask does not have the same number of points in time as the data")
        ratou = computes.compute_entropy_rate_ann_mask(&x[0,0], &mask[0], npts, nx, m, stride, 
                                    Theiler, N_eff, N_real, k, method, &S)
     else:
        ratou = computes.compute_entropy_rate_ann     (&x[0,0],           npts, nx, m, stride, 
                                    Theiler, N_eff, N_real, k, method, &S)
     return S



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_relative_entropy(double[:, ::1] x, double[:, ::1] y, int n_embed_x=1, int n_embed_y=1, int stride=1, 
                int Theiler=0, int N_eff=0, int N_real=0,
                int k=commons.k_default, int method=1):
     """      
     computes relative entropy of two processes (possibly multi-dimensional) using nearest neighbors search with ANN library.
     (time-)embedding is performed on the fly.
          
     :param x: signal (NumPy array with ndim=2, time along second dimension)
     :param y: signal (NumPy array with ndim=2, time along second dimension)
     :param n_embed_x: embedding dimension for x (default=1)
     :param n_embed_y: embedding dimension for y (default=1)
     :param stride: stride for embedding (default=1) 
     :param Theiler: Theiler scale (should be >= stride, but lower values are tolerated). If Theiler<0, then automatic Theiler is applied as described in function :any:`set_Theiler`.        
     :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
     :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
     :param k: number of neighbors to consider or -1 to force a non-ANN computation using covariance only, assuming Gaussian statistics.
     :param mask: masks are not supported yet.
     :param method: 0 for relative entropy (1 value [Hr] is returned) or 1 for Kullbach-Leibler divergence (2 values [Hr, KLdiv] are returned) (default=1)
  
     :returns: 1 or 2 values are returned, depending on the value of the parameter method: the relative entropy (Hr) and the KL divergence (Hr-H) estimates.
     
     see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.
     """

     cdef double Hr=0., H=0.
     cdef int npts=x.shape[1], nx=x.shape[0], npty=y.shape[1], ny=y.shape[0], ratou
     
     if (npts<nx):    raise ValueError("please transpose x")
     if (npty<ny):    raise ValueError("please transpose y")
     
     if (Theiler==0): Theiler=commons.samp_default.Theiler
     if (N_eff==0):   N_eff =commons.samp_default.N_eff
     if (N_real==0):  N_real=commons.samp_default.N_real
     
     ratou = computes.compute_relative_entropy_ann(&x[0,0], npts, &y[0,0], npty, nx, ny, n_embed_x, n_embed_y, stride, Theiler, N_eff, N_real, k, &Hr)
     if (method==0): return Hr
     
     cdef double std_Hr=0., std_H=0., tmp=0.
     commons.get_last_stds(&std_Hr, &tmp)
     ratou = computes.compute_entropy_ann(&x[0,0], npts, nx, n_embed_x, stride, Theiler, N_eff, N_real, k, &H)
     commons.get_last_stds(&std_H, &tmp)
     
     commons.set_last_stds(std_Hr, std_H)  # if not, returned std is the one for H, not Hr anymore!
     return [Hr, Hr-H]



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_MI(double[:, ::1] x, double[:, ::1] y, int n_embed_x=1, int n_embed_y=1, int stride=1, 
            int Theiler=0, int N_eff=0, int N_real=0,
            int k=commons.k_default, char[::1] mask=PNP.zeros(shape=(1),dtype='i1')):
     """ 
     computes mutual information MI(x,y) of two multi-dimensional vectors x and y using nearest neighbors search with ANN library.
     
     :param x, y: signals (NumPy arrays with ndim=2, time along second dimension)
     :param n_embed_x: embedding dimension for x (default=1)
     :param n_embed_y: embedding dimension for y (default=1)
     :param stride: stride for embedding (default=1)
     :param Theiler: Theiler scale (should be >= stride, but lower values are tolerated). If Theiler<0, then automatic Theiler is applied as described in function :any:`set_Theiler`.        
     :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
     :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
     :param k: number of neighbors to consider or -1 to force a non-ANN computation using covariance only, assuming Gaussian statistics.
     :param mask: mask to use (NumPy array of dtype=char). If a mask is provided, only values given by the mask will be used. (default=no mask)
                 
     :returns: two values (one per algorithm). See function :any:`choose_algorithm`
     
     see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.                 
     """
     cdef double I1=0, I2=0
     cdef int npts=x.shape[1], nx=x.shape[0], npty=y.shape[1], ny=y.shape[0]
     cdef int npts_mask=mask.size
     
     if (npts<nx):     raise ValueError("please transpose x")
     if (npty<ny):     raise ValueError("please transpose y")
     if (npty!=npts):  raise ValueError("x and y do not have same number of points in time")
     if (Theiler==0):  Theiler=commons.samp_default.Theiler
     if (N_eff==0):    N_eff =commons.samp_default.N_eff
     if (N_real==0):   N_real=commons.samp_default.N_real

     if (k==-1):
        ratou = computes.compute_mutual_information_Gaussian(&x[0,0], &y[0,0],
                                    npts, nx, ny, n_embed_x, n_embed_y, stride, Theiler, N_eff, N_real,    &I1) 
        return [I1, PNP.nan]  
               
     if (npts_mask>1): # then this is a real mask, not just the default value
        if (npts_mask!=npts): raise ValueError("mask's size doesn't match data's shape")
        ratou = computes.compute_mutual_information_ann_mask(&x[0,0], &y[0,0], &mask[0], 
                                    npts, nx, ny, n_embed_x, n_embed_y, stride, Theiler, N_eff, N_real, k, &I1, &I2)
     else:
        ratou = computes.compute_mutual_information_ann     (&x[0,0], &y[0,0], 
                                    npts, nx, ny, n_embed_x, n_embed_y, stride, Theiler, N_eff, N_real, k, &I1, &I2)
     return [I1,I2]



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_TE(double[:, ::1] x, double[:, ::1] y, int n_embed_x=1, int n_embed_y=1, int stride=1, 
            int Theiler=0, int N_eff=0, int N_real=0,
            int lag=1, int k=commons.k_default, char[::1] mask=PNP.zeros(shape=(1),dtype='i1')):
     """           
     computes the transfer entropy TE(x->y) (influence of x over y) of two n-d vectors x and y using nearest neighbors search with ANN library.
     
     TE(x,y) = TE(x->y)  (from x to y)
          
     :param x: (y): signals (NumPy arrays with ndim=2, time along second dimension)
     :param n_embed_x: embedding dimension for x (default=1)
     :param n_embed_y: embedding dimension for y (default=1)
     :param stride: stride for embedding (default=1)     
     :param lag: distance of the future point in time (default=1)
     :param Theiler: Theiler scale (should be >= stride, but lower values are tolerated). If Theiler<0, then automatic Theiler is applied as described in function :any:`set_Theiler`.        
     :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
     :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
     :param k: number of neighbors to consider or -1 to force a non-ANN computation using covariance only, assuming Gaussian statistics.
     :param mask: mask to use (NumPy array of dtype=char). If a mask is provided, only values given by the mask will be used. (default=no mask)
                 
     :returns: two values (one per algorithm)
     
     see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.
     """
     cdef double I1=0, I2=0
     cdef int npts=x.shape[1], nx=x.shape[0], npty=y.shape[1], ny=y.shape[0]
     cdef int npts_mask=mask.size, ratou
      
     if (npts<nx):    raise ValueError("please transpose x")
     if (npty<ny):    raise ValueError("please transpose y")
     if (npty!=npts): raise ValueError("x and y do not have the same number of pts in time")
     if (Theiler==0): Theiler=commons.samp_default.Theiler
     if (N_eff==0):   N_eff =commons.samp_default.N_eff
     if (N_real==0):  N_real=commons.samp_default.N_real
 
     if (k==-1):
            ratou = computes.compute_transfer_entropy_Gaussian(&x[0,0], &y[0,0], 
                                    npts, nx, ny, n_embed_x, n_embed_y, stride, lag, Theiler, N_eff, N_real, &I1)
            return [I1, PNP.nan]
         
#     get_sampling()
     if (mask.size>1): # then this is a real mask, not just the default value
            if (npts_mask!=npts): raise ValueError("mask's size doesn't match data's shape")
            ratou = computes.compute_transfer_entropy_ann_mask(&x[0,0], &y[0,0], &mask[0], 
                                    npts, nx, ny, n_embed_x, n_embed_y, stride, lag, Theiler, N_eff, N_real, k, &I1, &I2)
     else:
            ratou = computes.compute_transfer_entropy_ann(&x[0,0], &y[0,0], 
                                    npts, nx, ny, n_embed_x, n_embed_y, stride, lag, Theiler, N_eff, N_real, k, &I1, &I2)
     return [I1,I2]



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_PMI(double[:, ::1] x, double[:, ::1] y, double[:, ::1] z, 
            int n_embed_x=1, int n_embed_y=1, int n_embed_z=1, int stride=1, 
            int Theiler=0, int N_eff=0, int N_real=0,
            int k=commons.k_default, char[::1] mask=PNP.zeros(shape=(1),dtype='i1')):
     """            
     computes partial mutual information (PMI) of three 2-d vectors x, y and z using nearest neighbors search with ANN library
     
     PMI = MI(x,y|z)    (z is the conditioning variable)
          
     :param x, y, z: signals (NumPy arrays with ndim=2, time along second dimension)
     :param n_embed_x: embedding dimension for x (default=1)
     :param n_embed_y: embedding dimension for y (default=1)
     :param n_embed_z: embedding dimension for z (default=1)
     :param stride: stride for embedding (default=1)
     :param Theiler: Theiler scale (should be >= stride, but lower values are tolerated). If Theiler<0, then automatic Theiler is applied as described in function :any:`set_Theiler`.        
     :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
     :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
     :param k: number of neighbors to consider or -1 to force a non-ANN computation using covariance only, assuming Gaussian statistics.
     :param mask: mask to use (NumPy array of dtype=char). If a mask is provided, only values given by the mask will be used. (default=no mask)
                 
     :returns: two values (one per algorithm)
     
     see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.
     """
     cdef double I1=0, I2=0
     cdef int npts=x.shape[1], dim_x=x.shape[0], npty=y.shape[1], dim_y=y.shape[0]
     cdef int nptz=z.shape[1], dim_z=z.shape[0], npts_mask=mask.size
     cdef int dim[6]
     
     if (npts<dim_x):  raise ValueError("please transpose x")
     if (npty<dim_y):  raise ValueError("please transpose y")
     if (nptz<dim_z):  raise ValueError("please transpose z")          
     if (npty!=npts):  raise ValueError("x and y do not have the same the same number of points in time")
     if (nptz!=npts):  raise ValueError("x and z do not have the same the same number of points in time")
     if (Theiler==0):  Theiler=commons.samp_default.Theiler
     if (N_eff==0):    N_eff =commons.samp_default.N_eff
     if (N_real==0):   N_real=commons.samp_default.N_real
     
     dim[0]   = dim_x;      dim[1]   = dim_y;      dim[2]   = dim_z;
     dim[3+0] = n_embed_x;  dim[3+1] = n_embed_y;  dim[3+2] = n_embed_z;

     if (k==-1):
            ratou = computes.compute_partial_MI_Gaussian(&x[0,0], &y[0,0], &z[0,0], 
                            npts, dim, stride, Theiler, N_eff, N_real, &I1);
            return [I1, PNP.nan]  

     if (mask.size>1): # then this is a real mask, not just the default value
            if (npts_mask!=npts):  raise ValueError("mask's size doesn't match data's shape")
            ratou = computes.compute_partial_MI_ann_mask(&x[0,0], &y[0,0], &z[0,0], &mask[0],
                            npts, dim, stride, Theiler, N_eff, N_real, k, &I1, &I2);
     else:
            ratou = computes.compute_partial_MI_ann     (&x[0,0], &y[0,0], &z[0,0], 
                            npts, dim, stride, Theiler, N_eff, N_real, k, &I1, &I2);
     return [I1,I2]



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_PTE(double[:, ::1] x, double[:, ::1] y, double[:, ::1] z, 
            int n_embed_x=1, int n_embed_y=1, int n_embed_z=1, 
            int stride=1, int lag=1, 
            int Theiler=0, int N_eff=0, int N_real=0, 
            int k=commons.k_default, char[::1] mask=PNP.zeros(shape=(1),dtype='i1')):
     """           
     computes partial transfer entropy (PTE) of three 2-d vectors x, y and z using nearest neighbors search with ANN library.

     PTE(x,y,z) = TE(x->y|z)  (from x to y, with z a conditioning variable)
          
     :param x, y, z: signals (NumPy arrays with ndim=2, time along second dimension)
     :param n_embed_x: embedding dimension for x (default=1)
     :param n_embed_y: embedding dimension for y (default=1)
     :param n_embed_z: embedding dimension for z (default=1)
     :param stride: stride for embedding (default=1)     
     :param lag: distance of the future point in time (default=1)
     :param Theiler: Theiler scale (should be >= stride, but lower values are tolerated). If Theiler<0, then automatic Theiler is applied as described in function :any:`set_Theiler`.        
     :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
     :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
     :param k: number of neighbors to consider or -1 to force a non-ANN computation using covariance only, assuming Gaussian statistics.
     :param mask: mask to use (NumPy array of dtype=char). If a mask is provided, only values given by the mask will be used. (default=no mask)
                 
     :returns: two values (one per algorithm)
     
     see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.
     """
     cdef double I1=0, I2=0
     cdef int npts=x.shape[1], dim_x=x.shape[0], npty=y.shape[1], dim_y=y.shape[0]
     cdef int nptz=z.shape[1], dim_z=z.shape[0], npts_mask=mask.size
     cdef int dim[6]
     cdef int ratou
     
     if (npts<dim_x): raise ValueError("please transpose x")
     if (npty<dim_y): raise ValueError("please transpose y")
     if (nptz<dim_z): raise ValueError("please transpose z")        
     if (npty!=npts): raise ValueError("x and y do not have the same the same number of points in time")
     if (nptz!=npts): raise ValueError("x and z do not have the same the same number of points in time")
     if (Theiler==0): Theiler=commons.samp_default.Theiler
     if (N_eff==0):   N_eff =commons.samp_default.N_eff
     if (N_real==0):  N_real=commons.samp_default.N_real
    
     dim[0]   = dim_x;      dim[1]   = dim_y;      dim[2]   = dim_z;
     dim[3+0] = n_embed_x;  dim[3+1] = n_embed_y;  dim[3+2] = n_embed_z;

     if (mask.size>1): # then this is a real mask, not just the default value
            if (npts_mask!=npts):  raise ValueError("mask's size doesn't match data's shape")
     else:
#            mask = PNP.ones(shape=(1,npts), dtype='i1') # arbitrary convention #1
             mask = PNP.ones(npts, dtype='i1')           # simpler arbitrary convention 
     ratou = computes.compute_partial_TE_ann_mask(&x[0,0], &y[0,0], &z[0,0], &mask[0],
                            npts, dim, stride, lag, Theiler, N_eff, N_real, k, &I1, &I2);
     return [I1,I2]



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_DI(double[:, ::1] x, double[:, ::1] y, int N, int stride=1,
            int Theiler=0, int N_eff=0, int N_real=0,
            int k=commons.k_default, char[::1] mask=PNP.zeros(shape=(1),dtype='i1')):
     """       
     computes directed information DI(x->y) between two n-d vectors x and y, using nearest neighbors search with ANN library.
     (time-)embedding is performed on the fly.
          
     :param N: embedding dimension for x and y (should be >=1)
     :param stride: stride (Theiler correction will be used accordingly, even if n_embed_x,y=1). Note that for DI, lag (equivalent to stride for future point in time) is equal to stride.
     :param Theiler: Theiler scale (should be >= stride, but lower values are tolerated). If Theiler<0, then automatic Theiler is applied as described in function :any:`set_Theiler`.        
     :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
     :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
     :param k: number of neighbors to consider or -1 to force a non-ANN computation using covariance only, assuming Gaussian statistics.
     :param mask: mask to use (NumPy array of dtype=char). If a mask is provided, only values given by the mask will be used. (default=no mask)
                 
     :returns: two values (one per algorithm)
     
     see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.
     """
     cdef double I1=0, I2=0
     cdef int npts=x.shape[1], nx=x.shape[0], npty=y.shape[1], ny=y.shape[0]
     cdef int npts_mask=mask.size, ratou
     
     if (npts < nx):  raise ValueError("please transpose x")
     if (npty < ny):  raise ValueError("please transpose y")
     if (npty!=npts): raise ValueError("x and y do not have the same number of points in time")
     if (Theiler==0): Theiler=commons.samp_default.Theiler
     if (N_eff==0):   N_eff =commons.samp_default.N_eff
     if (N_real==0):  N_real=commons.samp_default.N_real
       
     if (k==-1):
         ratou = computes.compute_directed_information_Gaussian   (&x[0,0], &y[0,0],           npts, nx, ny, N, stride, Theiler, N_eff, N_real, &I1)
         return [I1, PNP.nan]
         
     if (mask.size>1): # then this is a real mask, not just the default value
            if (npts_mask!=npts):  raise ValueError("mask's size doesn't match data's shape")
            ratou = computes.compute_directed_information_ann_mask(&x[0,0], &y[0,0], &mask[0], npts, nx, ny, N, stride, Theiler, N_eff, N_real, k, &I1, &I2)
     else:  ratou = computes.compute_directed_information_ann     (&x[0,0], &y[0,0],           npts, nx, ny, N, stride, Theiler, N_eff, N_real, k, &I1, &I2)
     return [I1,I2]



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_regularity_index( double[:, ::1] x, int stride=1, 
            int Theiler=0, int N_eff=0, int N_real=0,
            int k=commons.k_default, char[::1] mask=PNP.zeros(shape=(1),dtype='i1')):
     """     
     computes regularity index of a vector (possibly multi-dimensional) using nearest neighbors search with ANN library.
     (time-)embedding is performed on the fly.
     
     Delta(x,tau) = H(\delta_tau x) - h^\tau(x) 
     
     :param x: signal (NumPy array with ndim=2, time as second dimension)
     :param stride: stride (Theiler correction will be used accordingly). 
     :param Theiler: Theiler scale (should be >= stride, but lower values are tolerated). If Theiler<0, then automatic Theiler is applied as described in function :any:`set_Theiler`.        
     :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
     :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
     :param k: number of neighbors to consider or -1 to force a non-ANN computation using covariance only, assuming Gaussian statistics.
     :param mask: mask to use (NumPy array of dtype=char). If a mask is provided, only values given by the mask will be used. (default=no mask)
                 
     :returns: two values (one per algorithm)
     
     see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.
     """
     cdef double D1=0, D2=0
     cdef int npts=x.shape[1], nx=x.shape[0], ratou 
     cdef int npts_mask=mask.size
     
     if (npts<nx):    raise ValueError("please transpose x")
     if (Theiler==0): Theiler=commons.samp_default.Theiler
     if (N_eff==0):   N_eff =commons.samp_default.N_eff
     if (N_real==0):  N_real=commons.samp_default.N_real
                   
     if (k==-1):
         ratou = computes.compute_regularity_index_Gaussian(&x[0,0],           npts, nx, 1, stride, Theiler, N_eff, N_real, &D1)
         return [D1, PNP.nan]           
                   
     if (npts_mask>1): # then this is a real mask, not just the default value
         if (npts_mask!=npts):
            raise ValueError("mask does not have the same number of points in time as the data")
#        ratou = computes.compute_regularity_index_ann_mask(&x[0,0], &mask[0], npts, nx, m, stride, k, method, &S)
         print("masks unsupported in this function, contact nicolas.garnier@ens-lyon.fr to add them")
     else:
         ratou = computes.compute_regularity_index_ann     (&x[0,0],           npts, nx, 1, stride, Theiler, N_eff, N_real, k, &D1, &D2)
     return [D1,D2]



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_entropy_2d( double[:, ::1] x, int n_embed=1, int stride_x=1, int stride_y=1, 
                int Theiler_x=0, int Theiler_y=0, int N_eff=0, int N_real=0,
                int k=commons.k_default, int method=0, int Theiler_2d=-1):
     """
     computes Shannon entropy of a scalar image, or of its spatial increments, using nearest neighbors search with ANN library.
     embedding and (uniform or random) sub-sampling are performed on the fly.
    
     :param x: signal (NumPy array with ndim=2 (unidimensional) or ndim=3 (multi-dimensional))
     :param stride_x: stride along x direction (Theiler correction will be used accordingly)  (default=1)
     :param stride_y: stride along y direction (Theiler correction will be used accordingly)  (default=1)
     :param Theiler_x: Theiler scale along x (should be >= stride_x, but lower values are tolerated).
     :param Theiler_y: Theiler scale along y (should be >= stride_x, but lower values are tolerated).
     :param N_eff: nb of points to consider in the statistics (default=4096) or -1 for largest possible value (legacy behavior)
     :param N_real: nb of realizations to consider (default=10) or -1 for N_real=stride (legacy behavior)
     :param k: number of neighbors to consider or -1 to force a non-ANN computation using covariance only, assuming Gaussian statistics.
     :param method: an interger in {0,1,2} with     
        0 for regular entropy of the image (default)
        1 for entropy of the increments (of the image)
        2 for entropy of the averaged increments (of the image)
     :param Theiler_2d: integers in {1, 2, 4} to specify which 2-d Theiler prescription to use.
     :returns: two values (one per algorithm)
     
     If either Theiler_x<0 or Theiler_y<0, then automatic Theiler is applied as follows:
        * -1 for Theiler=tau, and uniform sampling (thus localized in the dataset) (legacy)
        * -2 for Theiler=max, and uniform sampling (thus covering the all dataset)
        * -3 for Theiler=tau, and random sampling (default)
        * -4 for Theiler=max, and random sampling
        
     The parameter Theiler_2d indicates, depending on its value:
        * 1 for "minimal" : tau_Theiler is selected in each direction as in 1-d (troublesome if stride is small)
        * 2 for "maximal" : tau_Theiler is selected as the max of (stride_x, stride_y) (possible sqrt(2) trouble)
        * 4 for "optimal" : tau_Theiler is selected as sqrt(stride_x^2 + stride_y^2) (rounded-up)
        * if not provided, the value set by the function :any:`set_Theiler_2d` will be used (default="maximal").   
    
     see :any:`input_parameters` and function :any:`set_sampling` to set sampling parameters globally if needed.                
     """
     cdef double S=0
     cdef int d=1
     cdef int nx=x.shape[0], ny=x.shape[1], ratou 
     
     if (Theiler_x==0): Theiler_x=commons.samp_default.Theiler
     if (Theiler_y==0): Theiler_y=commons.samp_default.Theiler
     if (N_eff==0):     N_eff =commons.samp_default.N_eff
     if (N_real==0):    N_real=commons.samp_default.N_real
       
#     print("image", nx, "x", ny, "python function called with Theiler", Theiler_x, "/", Theiler_y)
     if (x.ndim==3):      # 2022/01/08: conventions have not been fixed yet
         d=x.shape[2]
         print("which dimensions correspond to the image, and which dimension corresponds to (multi-)components?")
         print("-> contact nicolas.garnier@ens-lyon.fr for discussing the conventions")
     # 2020-07-20: carefull with ordering of dimensions! Now, we use same as Python matrix!
    
     if (Theiler_2d>0): set_Theiler_2d(Theiler_2d) 
     ratou = computes.compute_entropy_ann_2d(&x[0,0], nx, ny, d, n_embed, stride_x, stride_y, Theiler_x, Theiler_y, N_eff, N_real, k, method, &S)
     return S

