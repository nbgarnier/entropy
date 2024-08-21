# entropy
an efficient C/C++ library integrated with Python and Matlab to estimate various entropies and many other quantities from information theory, using nearest neighbors estimates.

# compilation and installation
- run ./configure and eventuallly solve the issues by installing missing programs and libraries (e.g.: "apt install libtool-bin fftw3-dev" on Linux if asked to do so)
- then run either "make matlab" or "make python" (or both) to produce the library.
  
# Matlab version
- The Matlab version is fully supported as of 2023/10/09.
- "make matlab" should be working fine, as long as the configure step has correctly detected your matlab installation. To ensure the correct Matlab version is used (if you have several versions installed, or on MacOs if the auto-detection fails), re-run ./configure using the MATLAB option, e.g. on MacOs:
<pre><code>
./configure MATLAB=/Applications/MATLAB_R2023a.app
</code></pre>
- Matlab binaries and scripts are located in the subdirectory /bin/matlab ; you should add this path to your matlab environement in order to be able to run the functions provided by the library.

# Python version
- "make python" will both compile the library and install it in your python path, which depends on you current environment. You should select your environment first, then run "./configure" and "make python", in order to have the library and its functions available in your favored environment.
- there are examples in the bin/python subdirectory: please look at them to learn how to import and use the library, which should be as easy as:
<pre><code>
import numpy
import entropy.entropy as entropy

x = numpy.random.randn(1,100000)
H = entropy.compute_entropy(x)
</code></pre>

# notes

this library provides continuous entropies estimates using nearest neighbors. It relies on the [ANN library](http://www.cs.umd.edu/~mount/ANN/) by David Mount and Sunil Arya
