#####################################################################
# makefile for information theory functions                	    	#
#####################################################################
# 2022-06-01                                                        #
#####################################################################
# usages :                                                          #
#                                                                   #
# make C             : to produce C library                         #
# make python        : to produce python library                    #
# make matlab        : to produce matlab mex-files                  #
# make clean         : to remove all non-source files and clean     #
# make zip           : to produce an archive with source files only #
#####################################################################

#####################################################################
# edit the following file 'Make-config' to suit your needs          #
#####################################################################
include Make-config


#####################################################################
# following file contains Matlab definitions                        #
#####################################################################
include Make-Matlab-def


#####################################################################
# do not change anything below this line…                           #
#####################################################################
default:
	@echo "Enter one of the following:"
	@echo "  make C            for producing C library"
	@echo "  make matlab       for producing matlab mex files"
	@echo "  make python       for producing python library"
	@echo "  make cython       same as make python, but with re-cythonization"
	@echo "  make clean        for cleaning"
	@echo "  make zip          to produce code.tar.gz"
	@echo ""
	@echo "Note that the output/library will be stored under the bin/ directory,"
	@echo "in the relevant subdirectory. Please go there to use/copy the code !"

#####################################################################
# main make entry point                                             #
#####################################################################
C matlab python cython clean :
	cd src ; $(MAKE) $@ ; cd ..
#	cd src/cwt ; $(MAKE) $@ ; cd ../..


#####################################################################
# tarball                                                           #
#####################################################################
tarball : zip

zip : clean
ifeq ($(PLATFORM),MACOS)
	chmod -N *
endif
	$(TAR) -cvf code.tar Makefile Make-config Make-Matlab-def src/Makefile src/matlab/Make-Matlab
	$(TAR) -rvf code.tar include/
	$(TAR) -rvf code.tar src/ann_1.1.2/*
	$(TAR) -rvf code.tar src/ANN_wrapper.cpp src/ANN_wrapper_second_level.cpp \
		src/math_tools.c src/nns_count.c src/library_commons.c \
		src/mask.c src/samplings.c src/surrogates.c src/timings.c src/verbosity.c \
		src/entropy_ann_N.c \
		src/entropy_ann.c src/entropy_ann_threads.c src/entropy_ann_mask.c \
		src/entropy_ann_single_entropy.c src/entropy_ann_threads_entropy.c \
		src/entropy_ann_single_RE.c src/entropy_ann_threads_RE.c \
		src/entropy_ann_single_MI.c src/entropy_ann_threads_MI.c \
		src/entropy_ann_single_PMI.c src/entropy_ann_threads_PMI.c \
		src/entropy_ann_Renyi.c src/entropy_ann_threads_Renyi.c  \
		src/entropy_ann_combinations.c src/entropy_others.c \
		src/entropy_ann_2d.c \
		src/entropy_Gaussian.c src/entropy_Gaussian_single.c \
		src/entropy_Gaussian_combinations.c \
		src/main.c
	$(TAR) -rvf code.tar src/cython/setup.py src/cython/entropy/__init__.py \
		src/cython/entropy/entropy.pyx src/cython/entropy/entropy.c \
		src/cython/entropy/commons.pyx src/cython/entropy/commons.pxd \
		src/cython/entropy/computes.pyx src/cython/entropy/computes.pxd \
		src/cython/entropy/others.pyx src/cython/entropy/others.pxd \
		src/cython/entropy/tools.py src/cython/entropy/masks.py
	$(TAR) -rvf code.tar src/cython/tools/__init__.py src/cython/tools/tools.py \
		src/cython/tools/masks.py
	$(TAR) -rvf code.tar src/matlab/matlab_commons.c \
		src/matlab/compute_entropy.c src/matlab/compute_entropy_inc.c \
		src/matlab/compute_entropy_rate.c src/matlab/compute_regularity_index.c \
		src/matlab/compute_MI.c src/matlab/compute_TE.c \
		src/matlab/compute_PTE.c src/matlab/compute_PMI.c src/matlab/compute_DI.c \
		src/matlab/compute_Renyi.c src/matlab/compute_relative_entropy.c \
		src/matlab/compute_entropy_2d.c  \
		src/matlab/compute_ApEn.c src/matlab/compute_SampEn.c src/matlab/compute_KC.c 
	$(TAR) -rvf code.tar bin/matlab/compute_entropy.m bin/matlab/compute_entropy_inc.m \
		bin/matlab/compute_entropy_rate.m bin/matlab/compute_regularity_index.m \
		bin/matlab/compute_MI.m bin/matlab/compute_TE.m \
		bin/matlab/compute_PTE.m bin/matlab/compute_PMI.m bin/matlab/compute_DI.m \
		bin/matlab/compute_Renyi.m bin/matlab/compute_relative_entropy.m \
		bin/matlab/compute_entropy_2d.m \
		bin/matlab/compute_ApEn.m bin/matlab/compute_SampEn.m bin/matlab/compute_KC.m 
	$(TAR) -rvf code.tar bin/matlab/embed.m bin/matlab/compute_entropyrate_NaN.m \
		bin/matlab/compute_MI_NaN.m bin/matlab/compute_TE_NaN.m
	$(TAR) -rvf code.tar bin/python/*.py
	$(TAR) -rvf code.tar lib
	gzip code.tar
	mv code.tar.gz code-entropy-$(shell date +%Y-%m-%d).tar.gz
