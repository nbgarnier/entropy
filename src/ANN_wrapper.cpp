//----------------------------------------------------------------------
// File:			ANN_wrapper.cpp
//		
// Description:	header file for ANN_wrapper.c, to use ANN in C
// This is a C++ header
//
// lines 84 and 89 are important depending on the algo choice
//
// 2017-11-29: added function "search_ANN_external" (for relative entropy)
// 2017-11-29: to-do: simplify parameters call of "search_ANN"
// 2017-11-29: to-do: use global variable "dists" instead of re-computing epsilon in "search_ANN*"
//
// 2021-11-29: multithread adaptations started...
// 2021-12-01: multithread entropy OK
// 2025-02-13: added specific housekeeping for Relative Entropy engines
//----------------------------------------------------------------------
#define noDEBUG
#define DEBUG_N 37

#include "ANN/ANN.h"
#include "ANN_wrapper.h"        // definitions of functions only

#define UNUSED(expr) do { (void)(expr); } while (0)
// https://stackoverflow.com/questions/1486904/how-do-i-best-silence-a-warning-about-unused-variables


// #include "ann_1.1.2/src/pr_queue_k.h"	// 2021-12-01, k-element priority queue, for definition of ANNmin_k

// global variables for (internal, but lower level so "private") operations on the main tree:
// note that for these variables, moved from other .cpp files, the names have been kept for
// consistency accross the library
                                    
// 2021-12-03: allocation of these modified (for pthread) variables here in ANN_wrapper.cpp
// global variables for (internal, private or lower level) operations on the sub-trees:
// the following variables are defined "for real" in "kd_fix_rad_search_cpp" 
// and their names have been kept for consistency accross the library
//extern int	    *ANNkdFRDim;		// dimension of space
//extern ANNpoint *ANNkdFRQ;          // query point
//extern ANNdist  *ANNkdFRSqRad;		// squared radius search bound                                    
//extern ANNmin_k	**ANNkdFRPointMK;   // set of k closest points (unused)
//extern int	    *ANNkdFRPtsVisited; // total points visited (usused indeed)
//extern int	    *ANNkdFRPtsInRange; // number of points in the range // 2021-12-07: needs a mutex!!!
extern pthread_mutex_t mutex_annkFRSearch;     // 2021-12-07, NBG: experimentation
extern pthread_mutex_t mutex_strong;     // 2021-12-07, NBG: experimentation

// global variables for the main tree, internal to the C++ code, but exposed only in ANN_Wrapper.cpp:
double	        ANN_eps = 0.f;  // error bound, exact search if 0
ANNpointArray	dataPts;        // data points, type (*(*double))
ANNidxArray	   *nnIdx;          // k-nn indices    // 2021, adapted for pthread 
ANNdistArray   *dists;          // k-nn distances  // 2021, adapted for pthread
ANNkd_tree*	    kdTree;         // search structure

// global variables for the subtrees, internal to the C++ code, but exposed only in ANN_Wrapper.cpp:
ANNpointArray   dataPts_1;      // data points, type (*(*double))
ANNkd_tree*     kdTree_1;       // search structure
 
ANNpointArray   dataPts_2;      // data points, type (*(*double))
ANNkd_tree*     kdTree_2;       // search structure
 
ANNpointArray   dataPts_3;      // data points, type (*(*double))
ANNkd_tree*     kdTree_3;       // search structure

// allocate function (housekeeping) :
int init_ANN(int maxPts, int dim, int max_k, int NCORES=1)
{   if (NCORES<1) return(-1);

 	dataPts = annAllocPts(maxPts, dim);			// allocate data points
 	 	
	nnIdx = new ANNidx*[NCORES];				// allocate nearest neighbors indices 
	for (int i=0; i<NCORES; i++) nnIdx[i] = new ANNidx[max_k+1];
	
	dists = new ANNdist*[NCORES];               // allocate nearest neighbors distances
	for (int i=0; i<NCORES; i++) dists[i] = new ANNdist[max_k+1];
				
//    std::cout << "[init_ANN] OK, using " << NCORES <<" core(s)\n";
    return(0);
}



// allocate function (housekeeping) :
// 2025-02-13, new specific function for relative entropy (completely replace "init_ANN")
int init_ANN_RE(int maxPts_x, int maxPts_y, int dim, int max_k, int NCORES=1)
{   // int maxPts = (maxPts_x>maxPts_y) ? maxPts_x : maxPts_y;

    // the main tree is for y:
    init_ANN(maxPts_y, dim, max_k, NCORES); // or 2*NCORES?

    // a secondary tree is for x:
    dataPts_1 = annAllocPts(maxPts_x, dim);   
			
    return(0);
}



// allocate function (housekeeping) :
// 2019-01-30, version for MI, 2 arguments and 2 subspaces
// 2021-12-03, version for multithreading
int init_ANN_MI(int maxPts, int dim_1, int dim_2, int max_k, int NCORES=1)
{
    // the main tree:
    init_ANN(maxPts, dim_1+dim_2, max_k, 2*NCORES);
    
    // the subtrees:
    dataPts_1 = annAllocPts(maxPts, dim_1);     // allocate data points
    dataPts_2 = annAllocPts(maxPts, dim_2);     // allocate data points
	
    return(0);
}



// allocate function (housekeeping) :
// 2019-01-30, version for PMI et al, 3 arguments and 3 subspaces
int init_ANN_PMI(int maxPts, int dim_1, int dim_2, int dim_3, int max_k, int NCORES=1)
{
    // the main tree:
    init_ANN(maxPts, dim_1+dim_2+dim_3, max_k, 3*NCORES);
    
    // the subtrees:
    dataPts_1 = annAllocPts(maxPts, dim_1+dim_3);  // allocate data points
	dataPts_2 = annAllocPts(maxPts, dim_3);        // allocate data points
    dataPts_3 = annAllocPts(maxPts, dim_2+dim_3);  // allocate data points

    return(0);
}



// following two functions are new, to set the ANN_ALLOW_SELF_MATCH variable/singleton
// new 2014/02
// commented out 2014-06-03
/*
int set_ANN_state(char CHOICE)
{   int tmp;

    tmp = set_ANN_ALLOW_SELF_MATCH((char)CHOICE);
    std::cout << "set to " << tmp <<"\n";
    return(set_ANN_ALLOW_SELF_MATCH((char)CHOICE));
}

int get_ANN_state(void)
{   int tmp = get_ANN_ALLOW_SELF_MATCH();
    std::cout << "got " << tmp << "\n";
    return(tmp);
}
*/


// to do in create_kd_tree()
// dataPts must be allocated !!!
int create_kd_tree(double *x, int npts, int n)
{   int i,j;

    for (i=0; i<npts; i++)
    for (j=0; j<n;  j++)
        dataPts[i][j] = x[i + j*npts]; // dataPts is an ANNpointArray, so an array of ANNpoints
        
    // build search structure :
    kdTree = new ANNkd_tree(dataPts, npts, n);
    return(0);
}

// 2019-01-30: version for 2 variables
// dataPts_1 must be allocated !!!
int create_kd_tree_1(double *x, int npts, int dim_1)
{   int i,j;

    for (i=0; i<npts; i++)
    for (j=0; j<dim_1;  j++)
        dataPts_1[i][j] = x[i + j*npts];
 
    kdTree_1 = new ANNkd_tree(dataPts_1, npts, dim_1);
    return(0);
}

// 2019-01-30: version for 2 variables
// dataPts_2 must be allocated !!!
int create_kd_tree_2(double *x, int npts, int dim_2)
{   int i,j;

    for (i=0; i<npts; i++)
    for (j=0; j<dim_2;  j++)
        dataPts_2[i][j] = x[i + j*npts];
 
    kdTree_2 = new ANNkd_tree(dataPts_2, npts, dim_2);
    return(0);
}

// 2019-01-31: version for 3 variables
// dataPts_3 must be allocated !!!
int create_kd_tree_3(double *x, int npts, int dim_3)
{   int i,j;

    for (i=0; i<npts; i++)
    for (j=0; j<dim_3;  j++)
        dataPts_3[i][j] = x[i + j*npts];
 
    kdTree_3 = new ANNkd_tree(dataPts_3, npts, dim_3);
    return(0);
}



/****************************************************************************************/
/* below : piece of code to search for nearest neighbors of an external point           */
/*         using a previously computed kd-tree (with ANN library)                       */
/*         This function returns the marginal distances epsilon_z of the                */
/*         corresponding ball, to be used by the Kraskov et al. "second" algorithm      */
/*                                                                                      */
/* input parameters:                                                                    */
/* x    : is the query point (the one to search neighbors of)                           */
/* n    : is the dimension of vector space (dimension of x)                             */
/* k    : is the rank of the neighbor to search for                                     */
/* core : index of thread or core on which to operate                                   */
/* output parameters:                                                                   */
/* epsilon_z : n-dimensional vector of maximal 1-d distances from x                     */
/*                                                                                      */
/* 2019-01-28: first version                                                            */
/* 2019-12-17: unchanged, but tested against "search_ANN_external": much better!        */
/* 2021-12-02: threaded version (parameter "core" introduced)                           */
/****************************************************************************************/
int ANN_marginal_distances_ex(double *x, int n, int k, double *epsilon_z, int core)
{   double eps_local;
    int d, l;
    
    kdTree->annkSearch(                     // search
                     x,                     // query point
                     k+ANN_ALLOW_SELF_MATCH,// number of near neighbors (including or excluding central point)
                     nnIdx[core],           // nearest neighbors (returned)
                     dists[core],           // distance (returned)
                     ANN_eps,
                     core);
    
    for (d=0; d<n; d++)  // for each dimension
    {   epsilon_z[d] = 0.0;
        for (l=0; l<k+ANN_ALLOW_SELF_MATCH; l++) // loop over the neighbors
        {   // nnIdx[l] is the index of the l-th nearest neighbor
            eps_local = fabs(x[d] - dataPts[nnIdx[core][l]][d]); // distance du l-ieme voisin, dans la direction d
            if (eps_local>epsilon_z[d]) epsilon_z[d] = eps_local; // search the max in this direction
        }
    }
    // 2019-01-29: at this stage, there is a different epsilon in each dimension
    return(0);
} /* end of function "ANN_marginal_distances_ex" ****************************************/



/***************************************************************************************/
/* below : piece of code to search for nearest neighbors of a point                    */
/*         using a previously computed kd-tree (with ANN library)                      */
/* faster and memory efficient coding                                                  */
/*                                                                                     */
/* input parameters:                                                                   */
/* i    : is the index of the query point (the one to search neighbors of)             */
/* n    : is the dimension of vector space (dimension of x)                            */
/* k    : is the rank of the neighbor to search for                                    */
/* output parameters:                                                                  */
/* (returned) : distance of the k-nn neighbor from x                                   */
/*                                                                                     */
/* 2019-01-22 - first version                                                          */
/* 2021-12-01 - thread version                                                         */
/***************************************************************************************/
double ANN_find_distance_in(int i, int n, int k, int core) // 2021-12-01: core=0 if not multithread!
{   UNUSED(n);
    kdTree->annkSearch(            // search
                       dataPts[i], //queryPt,            // query point
                       k+ANN_ALLOW_SELF_MATCH,  // number of near neighbors (including or excluding central point)
                       nnIdx[core],                // nearest neighbors (returned)
                       dists[core],                // distance (returned)
                       ANN_eps,
                       core);
// 2021-12-13, test   
#ifdef DEBUG 
    if (i==DEBUG_N)
    {   std::cout << "\t[ANN_find_distance_in] with k=" << k+ANN_ALLOW_SELF_MATCH << "\n";
        std::cout << "\tnnIdx = [ ";
        for (int q=0; q<k+ANN_ALLOW_SELF_MATCH; q++) std::cout << nnIdx[core][q] << " ";
        std::cout << "]\n\tdists = [ ";
        for (int q=0; q<k+ANN_ALLOW_SELF_MATCH; q++) std::cout << dists[core][q] << " ";
        std::cout << "]\n";
    }
#endif    
    return( ((nnIdx[core][k-1+ANN_ALLOW_SELF_MATCH]<0) ? 0. : (double)dists[core][k-1+ANN_ALLOW_SELF_MATCH]) ); 
    // distance from central point
} /* end of function "ANN_find_distance_in" ****************************************/



/****************************************************************************************/
/* same as above, but using tree1                                                       */
/****************************************************************************************/
double ANN_find_distance_in_tree1(int i, int n, int k, int core) // 2021-12-01: core=0 if not multithread!
{   UNUSED(n);
    kdTree_1->annkSearch(            // search
                       dataPts_1[i], //queryPt,            // query point
                       k+ANN_ALLOW_SELF_MATCH,  // number of near neighbors (including or excluding central point)
                       nnIdx[core],                // nearest neighbors (returned)
                       dists[core],                // distance (returned)
                       ANN_eps,
                       core);

    return( (double)dists[core][k-1+ANN_ALLOW_SELF_MATCH]); // distance from central point
} /* end of function "ANN_find_distance_in_tree1" **************************************/



/***************************************************************************************/
/* below : piece of code to search for nearest neighbors of a point                    */
/*         using a previously computed kd-tree (with ANN library)                      */
/* faster and memory efficient coding                                                  */
/*                                                                                     */
/* input parameters:                                                                   */
/* x    : is a d-dimensional query point (the one to search neighbors of)              */
/* n    : is the dimension of vector space (dimension of x)                            */
/* k    : is the rank of the neighbor to search for                                    */
/* output parameters:                                                                  */
/* (returned) : distance of the k-nn from x                                            */
/*                                                                                     */
/* 2019-01-22 - first version                                                          */
/***************************************************************************************/
double ANN_find_distance_ex(double *x, int n, int k, int core)
{   UNUSED(n);
    //int d;
    /* composantes du point central : */
    //for (d=0; d<n; d++) queryPt[d] = x[d];
    
    kdTree->annkSearch(x, // queryPt,            // query point
                       k+ANN_ALLOW_SELF_MATCH,  // number of near neighbors (including or excluding central point)
                       nnIdx[core],                // nearest neighbors (returned)
                       dists[core],                // distance (returned)
                       ANN_eps);

    return((double)dists[core][k-1+ANN_ALLOW_SELF_MATCH]); // distance from central point
} /* end of function "ANN_find_distance_ex" ****************************************/



/**************************************************************************************
 * ANN_count_nearest_neighbors_nd_treeX
 * to count nearest neighbors of a point x0 in a ball of radius epsilon in n-dimensions
 *
 * x0 is the central point (of dimension d, the dimension of the tree)
 * epsilon is the ball radius
 *
 * 2019-01-29 : first version
 * 2021-12-02 : multithread-safe versions
 *************************************************************************************/
int ANN_count_nearest_neighbors_nd_tree1(double *x0, double epsilon, int core)
{   return(kdTree_1->annkFRSearch(x0,   // query point
                        epsilon,        // squared radius (same as radius for L^\infty norm)
                                        // algo 1 has condition "<epsilon", hence it requires a factor
                                        // ( correction = (1.-1./ANNnpts) from calling function)
                        0,              // (number of near neighbors to return), k=0 to search and count
                        NULL, // nnIdx_1[core],  // nearest neighbors (returned if !=NULL)
                        NULL, // dists_1[core],  // distance (returned if !=NULL)
                        ANN_eps,
                        core)); 
}
int ANN_count_nearest_neighbors_nd_tree2(double *x0, double epsilon, int core)
{   return(kdTree_2->annkFRSearch(x0,   // query point
                        epsilon,        // squared radius (same as radius for L^\infty norm)
                                        // algo 1 has condition "<epsilon", hence it requires a factor
                                        // ( correction = (1.-1./ANNnpts) from calling function)
                        0,              // (number of near neighbors to return), k=0 to search and count
                        NULL, // nnIdx_2[core],  // nearest neighbors (returned if !=NULL)
                        NULL, // dists_2[core],  // distance (returned if !=NULL)
                        ANN_eps,
                        core));
}
int ANN_count_nearest_neighbors_nd_tree3(double *x0, double epsilon, int core)
{   return(kdTree_3->annkFRSearch(x0,   // query point
                        epsilon,        // squared radius (same as radius for L^\infty norm)
                                        // algo 1 has condition "<epsilon", hence it requires a factor
                                        // ( correction = (1.-1./ANNnpts) from calling function)
                        0,              // (number of near neighbors to return), k=0 to search and count
                        NULL, // nnIdx_3[core],  // nearest neighbors (returned if !=NULL)
                        NULL, // dists_3[core],  // distance (returned if !=NULL)
                        ANN_eps,
                        core));
}



/***********************************************************************************************/
// free_functions (housekeeping) :
void free_ANN(int NCORES)
{   
//    delete [] ANNkdDim;
//    annDeallocPts(ANNkdQ);
//    delete [] ANNkdPointMK;
    
    for (int i=0; i<NCORES; i++) { delete nnIdx[i]; }   delete [] nnIdx;
    for (int i=0; i<NCORES; i++) { delete dists[i]; }   delete [] dists;
    
    annDeallocPts(dataPts);    delete kdTree;             // clean main tree
    
    annClose();                // done with ANN
}

// 2025-02-13, RE version with 2 subspaces
void free_ANN_RE(int NCORES)
{
    annDeallocPts(dataPts_1);    delete kdTree_1;
 
    free_ANN(NCORES); // or 2*NCORES if used, see above in "init_ANN_RE"
}

// free_function (housekeeping) :
// 2019-01-30, version with 2 subspaces
void free_ANN_MI(int NCORES)
{
    annDeallocPts(dataPts_1);    delete kdTree_1;
    annDeallocPts(dataPts_2);    delete kdTree_2;
    
    // clean main tree and other global stuff, and then close ANN
    free_ANN(2*NCORES);        // 2021-12-06 corrected  
}

// free_function (housekeeping) :
// 2019-01-31, version with 3 subspaces
// 2022-02-07, possible memory leak corrected
void free_ANN_PMI(int NCORES)
{
    annDeallocPts(dataPts_3);    delete kdTree_3;
    annDeallocPts(dataPts_2);    delete kdTree_2;
    annDeallocPts(dataPts_1);    delete kdTree_1;

    free_ANN(3*NCORES);   // cleaning main tree and other global stuff, and closing ANN
}
