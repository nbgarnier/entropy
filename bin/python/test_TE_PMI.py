import numpy
import entropy.entropy as entropy
import entropy.tools as tools

from time import time
from numpy.random import normal 


# parameters (change them to play)
ndim = 1        # dimensionality of x and y (usually 1, but you can play)
npoints = 100000
std  = 3.       # std of the white noises (x and y)

m_embed_x  = 1  # dimensionality of embedding of x
m_embed_y  = 1  # dimensionality of embedding of y


N_eff  = 4000   # the larger, the better the estimations
N_real = 10     # to compute the errorbar (as the std of the estimator)
entropy.set_sampling(4, N_eff=N_eff, N_real=N_real)


def compare_TE_PMI(x,y, lag=1):
    stride=lag  # I suggest to indicate to the code that the timescale we are studying is "lag" (will be used by Theiler)
       
    # TE version:
    t1=time()
    res=entropy.compute_TE(x, y, m_embed_x, m_embed_y, stride=stride, lag=lag)
    std=entropy.get_last_info()[0]
    t1=time()-t1
    print("TE  x->y =", res[0], "+/-", std , "\t(elapsed time :",t1, "s)")

    t1=time()
    res=entropy.compute_TE(y, x, m_embed_x, m_embed_y, stride=stride, lag=lag)
    std=entropy.get_last_info()[0]
    t1=time()-t1
    print("TE  y->x =", res[0], "+/-", std , "\t(elapsed time :",t1, "s)")

    xf=x[:,lag:].copy()
    yf=y[:,lag:].copy()
    xn=x[:,:-lag].copy()
    yn=y[:,:-lag].copy()

    # PMI version:
    t1=time()
    res=entropy.compute_PMI(yf, xn, yn, n_embed_x=1, n_embed_y=m_embed_y, n_embed_z=m_embed_x, stride=stride)
    std=entropy.get_last_info()[0]
    t1=time()-t1
    print("PMI x->y =", res[0], "+/-", std, "\t(elapsed time :",t1, "s)")

    t1=time()
    res=entropy.compute_PMI(xf, yn, xn, n_embed_x=1, n_embed_y=m_embed_x, n_embed_z=m_embed_y, stride=stride)
    std=entropy.get_last_info()[0]
    t1=time()-t1
    print("PMI y->x =", res[0], "+/-", std, "\t(elapsed time :",t1, "s)")

    # MI version:
    t1=time()
    res=entropy.compute_MI(yf, xn, n_embed_x=m_embed_x, n_embed_y=m_embed_y, stride=stride)
    std=entropy.get_last_info()[0]
    t1=time()-t1
    print("MI  x->y =", res[0], "+/-", std , "\t(elapsed time :",t1, "s)")
    
    t1=time()
    res=entropy.compute_MI(xf, yn, n_embed_x=m_embed_x, n_embed_y=m_embed_y, stride=stride)
    std=entropy.get_last_info()[0]
    t1=time()-t1
    print("MI  y->x =", res[0], "+/-", std, "\t(elapsed time :",t1, "s)")




# fist, un-correlated noises:
print()
print("normal distributions, uncorrelated,", npoints, "points,", ndim, "dimensions.")
x = normal(size=(ndim,npoints))*std
y = normal(size=(ndim,npoints))*std
compare_TE_PMI(x,y)

# fist, un-correlated noises:
shift=77
print()
print("normal distributions, correlated as y(t)=x(t-shift)+noise with shift =", shift)
y[:,shift:] = 1/2*x[:,:-shift] + 1/2*normal(size=(ndim,npoints-shift))
print("lag = 1, we should not see anything:")
compare_TE_PMI(x,y)
print("lag =",shift,", we should have a non-zero TE(x->y):")
compare_TE_PMI(x,y, shift)

