import numpy



# the following function returns True if there is a NaN somewhere in the data x,
# and returns False otherwise.
def no_NaN(x):
    """
    :meta private:
    """
    return numpy.isfinite(x).all()
   


# the following function returns a mask corresponding
# to NaN values in the data x
# the mask is of "char" type (int8), as required by Cython module (as of 2020-02-27)
# the mask is uni-dimensional (flatten), as required by Cython module (as of 2020-02-27)
def mask_NaN(x):
    """
    returns the mask corresponding to NaN values in the data x
    """
    mask_nan = numpy.isnan(x).astype('i1')
    return mask_clean(mask_nan)
    
    
    
# the following function returns a mask corresponding
# to finite values (not NaN and not infinite) in the data x
# the mask is of "char" type (int8), as required by Cython module (as of 2020-02-27)
# the mask is uni-dimensional (flatten), as required by Cython module (as of 2020-02-27)
# 2020-03-02: tests => this function is as fast (if not faster) than "entropy.mask_finite"
def mask_finite(x):
    """
    returns the mask corresponding to finite, correct values in the data x
    """
    mask_finite = numpy.isfinite(x).astype('i1')
    return mask_clean(mask_finite)



def mask_clean(x):
    """
    makes any nd-array a compatible mask for the code
    
    at any given time t, if the mask has a True value in one dimension, 
    then the resulting mask will also have a True value at that time t (AND logic)
    """
    y=x.astype('i1') 
    if (y.ndim>1): # ndarray with at least 2 dimensions: not good
        if (y.shape[0]>y.shape[1]):
            y=y.transpose()
        for j in numpy.arange(y.shape[0]-1):
            y[0,:] = y[0,:] * y[j+1,:]  # we put all information in the first dimension
        y = y[0,:]
    return y.flatten()



def retain_from_mask(x, mask):
    """
    returns a new nd-array from input nd-array x using input mask
    """
    y   = numpy.array(mask).astype('i1')
    ind = numpy.array(numpy.where(y>0))
#    print(ind.shape, ind.dtype)
    out = numpy.array(x)[:,ind].copy()
#    print(out.shape)
    out = numpy.array(x)[:,ind[0]].copy()
#    print(out.shape)
    return out
    