#####################################################################
# makefile for information theory functions                         #
#####################################################################
# file located in directory src/                                    #
#####################################################################
# last revision 2023-10-02                                          #
#####################################################################
# usages :                                                          #
#                                                                   #
# make               : to produce executable from C                 #
# make python        : to produce python library                    #
# make matlab        : to produce matlab mex-files                  #
# make clean         : to remove all non-source files and clean     #
# make zip           : to produce an archive with source files only #
#####################################################################


# where is the root of the complete package :
BASE_DIR = ..
# where to put matlab and python libraries
OUTPUT_DIR = $(BASE_DIR)/bin
# where to put the C library
LIB_DIR = $(BASE_DIR)/lib
# where are Python definitions and source files for Cython version :
CYTHON_DIR =$(BASE_DIR)/src/cython


#-----------------------------------------------------------------------------
# inherit definitions from ./configure step
#-----------------------------------------------------------------------------
PLATFORM = MACOS
CC = gcc
CXX = g++
CFLAGS = -O3 -fPIC -std=c99 -pthread
CXXFLAGS = -O3 -fPIC -pthread
CPPFLAGS = -pedantic -Wall -Wextra -Wshadow -Wpointer-arith -Wundef -Wunreachable-code
CPPFLAGS += -I$(BASE_DIR)/include
PYTHON = python
LDLIBS = -lfftw3 -lgsl -lm  -lANN

ifeq ($(PLATFORM),LINUX)
	LDLIBS += -pthread
endif

# common C objects:
OBJS = ANN_wrapper.o ANN_wrapper_second_level.o \
	samplings.o surrogates.o mask.o library_commons.o verbosity.o \
	math_tools.o nns_count.o timings.o \
	entropy_ann.o entropy_ann_mask.o entropy_ann_Renyi.o \
	entropy_ann_single_entropy.o entropy_ann_single_RE.o \
	entropy_ann_single_MI.o entropy_ann_single_PMI.o \
	entropy_ann_threads.o entropy_ann_threads_entropy.o \
	entropy_ann_threads_Renyi.o entropy_ann_threads_RE.o\
	entropy_ann_threads_MI.o entropy_ann_threads_PMI.o \
	entropy_ann_combinations.o entropy_others.o \
	entropy_ann_2d.o entropy_ann_N.o \
	entropy_Gaussian.o entropy_Gaussian_single.o entropy_Gaussian_combinations.o

LDFLAGS += -L$(LIB_DIR)
MEX_LDFLAGS += -L$(LIB_DIR)

.PHONY : all C python cython matlab clean zip

C : $(LIB_DIR)/libentropy.a $(OUTPUT_DIR)/run.exe 


#####################################################################
# static libraries                                                  #
#####################################################################
#noinst_LIBRARIES = libANN.a


#####################################################################
# main C stuff                                                      #
#####################################################################
$(LIB_DIR)/libentropy.a: $(OBJS)
	libtool -arch_only x86_64 -static -o libentropy.a $(OBJS)
#	$(RANLIB) libentropy.a
	mv libentropy.a $(LIB_DIR)

$(OUTPUT_DIR)/run.exe : $(OBJS) main.o 
	$(CXX) $(OBJS) main.o $(LDLIBS) -o $(OUTPUT_DIR)/run.exe -L$(LIB_DIR)

main.o : main.c $(INCS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c main.c

#####################################################################
# ANN library and wrapper                                           #
#####################################################################
$(LIB_DIR)/libANN.a : $(BASE_DIR)/include/ANN/ANN.h
	cd ann_1.1.2/src ; $(MAKE) ; mv ../lib/libANN.a ../../$(LIB_DIR)/

#../lib/libANN.so : $(INCLUDE)/ANN/ANN.h
#	cd ann_1.1.2/src ; $(MAKE) shared; mv ../lib/libANN.so ../../$(LIB_DIR)/

$(BASE_DIR)/include/ANN/ANN.h :
	cp ann_1.1.2/include/ANN/*.h $(BASE_DIR)/include/ANN/

ANN_wrapper.o : $(BASE_DIR)/include/ANN/ANN.h $(BASE_DIR)/lib/libANN.a ANN_wrapper.cpp $(BASE_DIR)/include/ANN_wrapper.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c ANN_wrapper.cpp

ANN_wrapper_second_level.o : $(BASE_DIR)/include/ANN/ANN.h $(BASE_DIR)/lib/libANN.a ANN_wrapper_second_level.cpp $(BASE_DIR)/include/ANN_wrapper.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c ANN_wrapper_second_level.cpp


#####################################################################
# C objects:                                                        #
#####################################################################
entropy_ann.o : entropy_ann.c entropy_ann_threads.c $(BASE_DIR)/include/entropy_ann.h $(BASE_DIR)/include/nns_count.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $<

%.o : %.c $(BASE_DIR)/include/%.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $<


#####################################################################
# Python and Cython stuff                                           #
# Warning : the library will automatically be installed in the      #
# user Python library directory (so no need to copy it manually)    #
#####################################################################
python : $(LIB_DIR)/libANN.a $(LIB_DIR)/libentropy.a
#	cd cython; CC=$(CC) $(PYTHON) setup.py install --user; cd ..
	cd cython; CC=$(CC) pip install --user . -v -v -v; cd ..

cython : $(LIB_DIR)/libANN.a $(LIB_DIR)/libentropy.a cython/entropy/entropy.pyx cython/setup.py
	rm -f cython/entropy/entropy.c
	cd cython; CC=$(CC) pip install --user -v -v -v .; cd ..

#####################################################################
# Matlab stuff                                                      #
# (http://www.mathworks.fr/help/techdoc/matlab_external/f24338.html)#
# first step : create objects files with matlab options (.mo)       #
# second step : linking                                             #
#####################################################################
matlab : $(LIB_DIR)/libANN.a $(LIB_DIR)/libentropy.a
	cd matlab; $(MAKE); cd ..
	
		

#####################################################################
# rules to clean all intermediate and output files                  #
#####################################################################
clean :
	rm -f *.o *.a *.exe *.mo ../code.tar.gz
	cd $(BASE_DIR) ; rm -f -R ._* .DS_Store ; cd src/
# 2015-10-06, NG, added line above for MacOs garbage
	cd matlab ; make clean ; cd ..
	\rm -f -R build/
	rm -f $(OUTPUT_DIR)/run.exe
	rm -f ../lib/libANN.a ../lib/libentropy.a ../include/ANN/ANN.h
	cd ann_1.1.2/src ; make clean ; cd ../..
	cd cython ; $(PYTHON) setup.py clean --a ; cd ..
#	rm -f cython/entropy/entropy.c

