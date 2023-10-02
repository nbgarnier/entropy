# entropy
an efficient C/C++ library integrated with Python and Matlab to estimate various entropies and many other quantities from information theory, using nearest neighbors estimates.

# compilation and installation
- run ./configure and eventuallly solve the issues by installing missing programs and libraries
- then run either "make matlab" or "make python" to produce the library.
  
# Matlab version
- "make matlab" may not be working since last updates. The Matlab version is not fully supported as of now. Open an issue if this matters.

# Python version
- "make python" will both compile the library and install it in your python path.
- there are examples in the bin/python subdirectory: please look at them to learn how to import and use the library.
