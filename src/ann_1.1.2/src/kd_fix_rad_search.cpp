//----------------------------------------------------------------------
// File:			kd_fix_rad_search.cpp
// Programmer:		Sunil Arya and David Mount
// Description:		Standard kd-tree fixed-radius kNN search
// Last modified:	05/03/05 (Version 1.1)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
// 
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
// 
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 1.1  05/03/05
//		Initial release
//----------------------------------------------------------------------

#include "kd_fix_rad_search.h"			// kd fixed-radius search decls
#include <pthread.h>

#define UNUSED(expr) do { (void)(expr); } while (0)
// https://stackoverflow.com/questions/1486904/how-do-i-best-silence-a-warning-about-unused-variables


//----------------------------------------------------------------------
//	Approximate fixed-radius k nearest neighbor search
//		The squared radius is provided, and this procedure finds the
//		k nearest neighbors within the radius, and returns the total
//		number of points lying within the radius.
//
//		The method used for searching the kd-tree is a variation of the
//		nearest neighbor search used in kd_search.cpp, except that the
//		radius of the search ball is known.  We refer the reader to that
//		file for the explanation of the recursive search procedure.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//		To keep argument lists short, a number of global variables
//		are maintained which are common to all the recursive calls.
//		These are given below.
//
// 		2021-12-03, NBG: unfortunately, these global declarations make the 
// 		search being not thread-safe! 
//		Tentative corrections are started to be applied... 
//		which led to a modification of global variables
//		(some moved to "ANN_wrapper.cpp")
//
//		note that because functions here are used by both MI algorithm,
//		and we want to compute both estimates simultaneously, we need 
// 		twice as more threads. This should be done properly by declaring
//		a (twice as large) number of threads, while keeping good track 
//		of threads identities (via variable "core")
//----------------------------------------------------------------------

__thread int	   ANNkdFRDim;			// dimension of space
__thread ANNpoint  ANNkdFRQ;			// query point // 2021-12-03: adapted for pthread
__thread ANNdist ANNkdFRSqRad;			// squared radius search bound
__thread double	 ANNkdFRMaxErr;			// max tolerable squared error (thread-safe)
__thread ANNpointArray	ANNkdFRPts;		// the points                  (not thread-safe)
__thread ANNmin_k *ANNkdFRPointMK;		// set of k closest points 
__thread int	ANNkdFRPtsVisited;		// total points visited // not thread-safe!
__thread int	ANNkdFRPtsInRange;		// number of points in the range

//pthread_mutex_t mutex_annkFRSearch;   // 2021-12-07, NBG: experimentation
//pthread_mutex_t mutex_strong;     	// 2021-12-07, NBG: experimentation (remove pthread interest!)

#define DEBUG_string_size 8192
char 	DEBUG_string[DEBUG_string_size];
//----------------------------------------------------------------------
//	annkFRSearch - fixed radius search for k nearest neighbors
//----------------------------------------------------------------------

int ANNkd_tree::annkFRSearch(
	ANNpoint			q,				// the query point
	ANNdist				sqRad,			// squared radius search bound
	int					k,				// number of near neighbors to return
	ANNidxArray			nn_idx,			// nearest neighbor indices (returned)
	ANNdistArray		dd,				// the approximate nearest neighbor (returned?)
	double				eps,			// the error bound
	int					core)			// added 2021-12-01: an index to make call thread-safe
{
	UNUSED(nn_idx);
	UNUSED(dd);
	
//pthread_mutex_lock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation
	ANNkdFRDim = dim;				// copy arguments to static equivs
	ANNkdFRQ = q; // original code
//	2021-12-03 ANNkdFRQ was a unique global variable, now a global array, with one component per thread
//	for (register int j=0; j<dim; j++) ANNkdFRQ[core][j] = q[j];
	                                    // 2021-12-03: note: second copy of query => optimizable
	ANNkdFRSqRad = sqRad;
	ANNkdFRPts = pts;
	ANNkdFRPtsVisited = 0;		// initialize count of points visited
	ANNkdFRPtsInRange = 0;		// ...and points in the range
//pthread_mutex_unlock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation

//	sprintf(DEBUG_string, "%s", "");	// reset debug string
	DEBUG_string[0] = '\0';				// 2022-05-23: better
	
	ANNkdFRMaxErr = ANN_POW(1.0 + eps);
	ANN_FLOP(2)							// increment floating op count
//pthread_mutex_lock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation
	ANNkdFRPointMK = new ANNmin_k(k);	// create set for closest k points
//pthread_mutex_unlock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation	

										// search starting at the root
	root->ann_FR_search(annBoxDistance(q, bnd_box_lo, bnd_box_hi, dim), core);

// 2021-12-07, NBG: in the context of MI computation, we do not need (or use)
// the nearest neighbors themselves (either their indices or their distances)
// so I comment out this piece of code that returns these informations
/*	for (int i = 0; i < k; i++) {		// extract the k-th closest points
		if (dd != NULL)
			dd[i] = ANNkdFRPointMK[core]->ith_smallest_key(i);
		if (nn_idx != NULL)
			nn_idx[i] = ANNkdFRPointMK[core]->ith_smallest_info(i);
	}
*/
	delete ANNkdFRPointMK;		// deallocate closest point set
	
	return ANNkdFRPtsInRange;		// return final point count 
		// 2021-12-03 this return value should be made thread-safe!
}

//----------------------------------------------------------------------
//	kd_split::ann_FR_search - search a splitting node
//		Note: This routine is similar in structure to the standard kNN
//		search.  It visits the subtree that is closer to the query point
//		first.  For fixed-radius search, there is no benefit in visiting
//		one subtree before the other, but we maintain the same basic
//		code structure for the sake of uniformity.
//----------------------------------------------------------------------

void ANNkd_split::ann_FR_search(ANNdist box_dist, int core)
{
										// check dist calc term condition
	if (ANNmaxPtsVisited != 0 && ANNkdFRPtsVisited > ANNmaxPtsVisited) return;

										// distance to cutting plane
//pthread_mutex_lock(&mutex_annkFRSearch); // 2021-12-08, NBG: experimentation with cut_val
	ANNcoord cut_diff = ANNkdFRQ[cut_dim] - cut_val;
//pthread_mutex_unlock(&mutex_annkFRSearch); // 2021-12-08, NBG: experimentation

	if (cut_diff < 0) {					// left of cutting plane
		child[ANN_LO]->ann_FR_search(box_dist, core);// visit closer child first

//pthread_mutex_lock(&mutex_annkFRSearch); // 2021-12-08, NBG: experimentation with cut_dim
		ANNcoord box_diff = cd_bnds[ANN_LO] - ANNkdFRQ[cut_dim];
//pthread_mutex_unlock(&mutex_annkFRSearch); // 2021-12-08, NBG: experimentation with cut_dim		
		if (box_diff < 0)				// within bounds - ignore
			box_diff = 0;
										// distance to further box
		box_dist = (ANNdist) ANN_SUM(box_dist,
				ANN_DIFF(ANN_POW(box_diff), ANN_POW(cut_diff)));

										// visit further child if in range
		if (box_dist * ANNkdFRMaxErr <= ANNkdFRSqRad)
			child[ANN_HI]->ann_FR_search(box_dist, core);
	}
	else {								// right of cutting plane
		child[ANN_HI]->ann_FR_search(box_dist, core);// visit closer child first

//pthread_mutex_lock(&mutex_annkFRSearch); // 2021-12-08, NBG: experimentation with cut_dim
		ANNcoord box_diff = ANNkdFRQ[cut_dim] - cd_bnds[ANN_HI];
//pthread_mutex_unlock(&mutex_annkFRSearch); // 2021-12-08, NBG: experimentation with cut_dim		
		if (box_diff < 0)				// within bounds - ignore
			box_diff = 0;
										// distance to further box
		box_dist = (ANNdist) ANN_SUM(box_dist,
				ANN_DIFF(ANN_POW(box_diff), ANN_POW(cut_diff)));

										// visit further child if close enough
		if (box_dist * ANNkdFRMaxErr <= ANNkdFRSqRad)
			child[ANN_LO]->ann_FR_search(box_dist, core);

	}
	ANN_FLOP(13)						// increment floating ops
	ANN_SPL(1)							// one more splitting node visited
}

//----------------------------------------------------------------------
//	kd_leaf::ann_FR_search - search points in a leaf node
//		Note: The unreadability of this code is the result of
//		some fine tuning to replace indexing by pointer operations.
//----------------------------------------------------------------------

void ANNkd_leaf::ann_FR_search(ANNdist box_dist, int core)
{
	ANNdist dist;				// distance to data point
	ANNcoord* pp;				// data coordinate pointer
	ANNcoord* qq;				// query coordinate pointer
	ANNcoord t;
	int d;
	
	UNUSED(box_dist);
	UNUSED(core);
	
	for (int i = 0; i < n_pts; i++) {	// check points in bucket

//pthread_mutex_lock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation
		pp = ANNkdFRPts[bkt[i]];		// first coord of next data point
		qq = ANNkdFRQ;			// first coord of query point
//pthread_mutex_unlock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation		
		dist = 0;

		for(d = 0; d < ANNkdFRDim; d++) {
			ANN_COORD(1)				// one more coordinate hit
			ANN_FLOP(5)					// increment floating ops

//pthread_mutex_lock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation
			t = *(qq++) - *(pp++);		// compute length and adv coordinate
//pthread_mutex_unlock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation			
										// exceeds dist to k-th smallest?
			if( (dist = ANN_SUM(dist, ANN_POW(t))) > ANNkdFRSqRad) 
			{//	std::cout << "core " << core << "d=" << d << "ANNkdFRDim[core] = " << ANNkdFRDim[core] << "  dist " << dist << " vs epsilon = " << ANNkdFRSqRad[core] << "\n";
//				snprintf(DEBUG_string, DEBUG_string_size, "%score %d  d=%d  ANNkdFRDim[core] : %d  dist : %f  vs epsilon %f\n", DEBUG_string, core, d, ANNkdFRDim, dist, ANNkdFRSqRad);
				break;
			}
		}

		if (d >= ANNkdFRDim &&			// among the k best?
		   (ANN_ALLOW_SELF_MATCH || dist!=0)) { // and no self-match problem
												// add it to the list
				ANNkdFRPointMK->insert(dist, bkt[i]);
				ANNkdFRPtsInRange++;			// increment point count
		}
	}
	ANN_LEAF(1)							// one more leaf node visited
	ANN_PTS(n_pts)						// increment points visited
//	pthread_mutex_lock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation
	ANNkdFRPtsVisited += n_pts;	// increment number of points visited
//	pthread_mutex_unlock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation
}
