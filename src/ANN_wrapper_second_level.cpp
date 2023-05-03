//----------------------------------------------------------------------
// File:			ANN_wrapper_second_level.cpp
//		
// Description:	header file for ANN_wrapper.c, to use ANN in C
// This is a C++ header
//
// this file is for parallel invocation of the library, ie, concurrent calls to engine functions
//
// 2022-11-30: file forked from "ANN_wrapper.cpp"
//----------------------------------------------------------------------
#define noDEBUG
#define DEBUG_N 37

#include "ANN/ANN.h"
#include "ANN_wrapper_second_level.h"        // definitions of functions only

#define UNUSED(expr) do { (void)(expr); } while (0)
// https://stackoverflow.com/questions/1486904/how-do-i-best-silence-a-warning-about-unused-variables

// global variables for the main tree, internal to the C++ code, but exposed only in ANN_Wrapper_second_level.cpp:
extern double   ANN_eps;		// error bound, exact search if 0 // define for real in ANN_wrapper.cpp
ANNpointArray	dataPts_sl;     // data points, type (*(*double))
ANNidxArray	   *nnIdx_sl;       // k-nn indices    // 2021, adapted for pthread 
ANNdistArray   *dists_sl;       // k-nn distances  // 2021, adapted for pthread
ANNkd_tree*	    kdTree_sl;      // search structure

// global variables for the subtrees, internal to the C++ code, but exposed only in ANN_Wrapper.cpp:
ANNpointArray   dataPts_1_sl;      // data points, type (*(*double))
// ANNidxArray    *nnIdx_1;        // near neighbor indices
// ANNdistArray   *dists_1;        // near neighbor distances
ANNkd_tree*     kdTree_1_sl;       // search structure
 
ANNpointArray   dataPts_2_sl;      // data points, type (*(*double))
// ANNidxArray    *nnIdx_2;        // near neighbor indices
// ANNdistArray   *dists_2;        // near neighbor distances
ANNkd_tree*     kdTree_2_sl;       // search structure
 
ANNpointArray   dataPts_3_sl;      // data points, type (*(*double))
// ANNidxArray    *nnIdx_3;        // near neighbor indices
// ANNdistArray   *dists_3;        // near neighbor distances
ANNkd_tree*     kdTree_3_sl;       // search structure

// allocate function (housekeeping) :
int init_ANN_sl(int maxPts, int dim, int max_k, int NCORES=1)
{   if (NCORES<1) return(-1);

//    ANNkdDim = new int[NCORES];                 // 2021-12-07
//    ANNkdQ   = annAllocPts(NCORES, dim);        // 2021-12-01, OK
//    ANNkdPointMK = new ANNmin_k*[NCORES];       // 2021-12-01, new for pthread

 	dataPts_sl = annAllocPts(maxPts, dim);			// allocate data points
 	 	
	nnIdx_sl = new ANNidx*[NCORES];				// allocate nearest neighbors indices 
	for (int i=0; i<NCORES; i++) nnIdx_sl[i] = new ANNidx[max_k+1];
	
	dists_sl = new ANNdist*[NCORES];               // allocate nearest neighbors distances
	for (int i=0; i<NCORES; i++) dists_sl[i] = new ANNdist[max_k+1];
				
//    std::cout << "[init_ANN] OK, using " << NCORES <<" core(s)\n";
    return(0);
}


// allocate function (housekeeping) :
// 2019-01-30, version for MI, 2 arguments and 2 subspaces
// 2021-12-03, version for multithreading
int init_ANN_MI_sl(int maxPts, int dim_1, int dim_2, int max_k, int NCORES=1)
{
    // the main tree:
    init_ANN_sl(maxPts, dim_1+dim_2, max_k, 2*NCORES);
    
    // 2021-12-03, new for pthread: additional allocations for marginal counting 
    // (variables defined for real in kd_fix_rad_search.cpp):
    // note the factor 2 to account for the use by 2 algorithms simultaneously!
//    ANNkdFRDim        = new int[2*NCORES];
//  ANNkdFRQ          = annAllocPts(2*NCORES, dim_1+dim_2);  // 2021-12-03, check dim_1+dim_2 ?
//    ANNkdFRSqRad      = new ANNdist[2*NCORES];
//    ANNkdFRPointMK    = new ANNmin_k*[2*NCORES]; // next allocation step is done in the tree by annkFRSearch
//    ANNkdFRPtsVisited = new int[2*NCORES];
//    ANNkdFRPtsInRange = new int[2*NCORES];
//    pthread_mutex_init(&mutex_annkFRSearch, 0);     // 2021-12-07, NBG: experimentation
//    pthread_mutex_init(&mutex_strong, 0);           // 2021-12-07, NBG: experimentation
    
    // the subtrees:
// 2021-12-03 note: I think these variables are unused!!!     
// 2021-12-07 note: checked, they are indeed unused.   
    dataPts_1_sl = annAllocPts(maxPts, dim_1);     // allocate data points
//    nnIdx_1 = new ANNidx*[NCORES];              // allocate nearest neighbors indices 
//	  for (int i=0; i<NCORES; i++) nnIdx_1[i] = new ANNidx[max_k+1];
//    dists_1 = new ANNdist*[NCORES];             // allocate nearest neighbors distances
//	  for (int i=0; i<NCORES; i++) dists_1[i] = new ANNdist[max_k+1];
	
    dataPts_2_sl = annAllocPts(maxPts, dim_2);     // allocate data points
//     nnIdx_2 = new ANNidx*[NCORES];				// allocate nearest neighbors indices 
// 	  for (int i=0; i<NCORES; i++) nnIdx_2[i] = new ANNidx[max_k+1];
// 	dists_2 = new ANNdist*[NCORES];             // allocate nearest neighbors distances
// 	  for (int i=0; i<NCORES; i++) dists_2[i] = new ANNdist[max_k+1];
	
    return(0);
}

// allocate function (housekeeping) :
// 2019-01-30, version for PMI et al, 3 arguments and 3 subspaces
int init_ANN_PMI(int maxPts, int dim_1, int dim_2, int dim_3, int max_k, int NCORES=1)
{
    // the main tree:
    init_ANN_sl(maxPts, dim_1+dim_2+dim_3, max_k, 3*NCORES);
    
    // the subtrees:
    dataPts_1_sl = annAllocPts(maxPts, dim_1+dim_3);  // allocate data points
	dataPts_2_sl = annAllocPts(maxPts, dim_3);        // allocate data points
    dataPts_3_sl = annAllocPts(maxPts, dim_2+dim_3);  // allocate data points

    return(0);
}


// to do in create_kd_tree()
// dataPts must be allocated !!!
int create_kd_tree_sl(double *x, int npts, int n)
{   int i,j;

    for (i=0; i<npts; i++)
    for (j=0; j<n;  j++)
        dataPts_sl[i][j] = x[i + j*npts]; // dataPts is an ANNpointArray, so an array of ANNpoints
        
    // build search structure :
    kdTree_sl = new ANNkd_tree(dataPts_sl, npts, n);
    return(0);
}

// 2019-01-30: version for 2 variables
// dataPts_1 must be allocated !!!
int create_kd_tree_1_sl(double *x, int npts, int dim_1)
{   int i,j;

    for (i=0; i<npts; i++)
    for (j=0; j<dim_1;  j++)
        dataPts_1_sl[i][j] = x[i + j*npts];
 
    kdTree_1_sl = new ANNkd_tree(dataPts_1_sl, npts, dim_1);
    return(0);
}

// 2019-01-30: version for 2 variables
// dataPts_2 must be allocated !!!
int create_kd_tree_2_sl(double *x, int npts, int dim_2)
{   int i,j;

    for (i=0; i<npts; i++)
    for (j=0; j<dim_2;  j++)
        dataPts_2_sl[i][j] = x[i + j*npts];
 
    kdTree_2_sl = new ANNkd_tree(dataPts_2_sl, npts, dim_2);
    return(0);
}

// 2019-01-31: version for 3 variables
// dataPts_3 must be allocated !!!
int create_kd_tree_3_sl(double *x, int npts, int dim_3)
{   int i,j;

    for (i=0; i<npts; i++)
    for (j=0; j<dim_3;  j++)
        dataPts_3_sl[i][j] = x[i + j*npts];
 
    kdTree_3_sl = new ANNkd_tree(dataPts_3_sl, npts, dim_3);
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
int ANN_marginal_distances_ex_sl(double *x, int n, int k, double *epsilon_z, int core)
{   double eps_local;
    int d, l;
    
    kdTree_sl->annkSearch(                     // search
                     x,                     // query point
                     k+ANN_ALLOW_SELF_MATCH,// number of near neighbors (including or excluding central point)
                     nnIdx_sl[core],           // nearest neighbors (returned)
                     dists_sl[core],           // distance (returned)
                     ANN_eps,
                     core);
    
    for (d=0; d<n; d++)  // for each dimension
    {   epsilon_z[d] = 0.0;
        for (l=0; l<k+ANN_ALLOW_SELF_MATCH; l++) // loop over the neighbors
        {   // nnIdx[l] is the index of the l-th nearest neighbor
            eps_local = fabs(x[d] - dataPts_sl[nnIdx_sl[core][l]][d]); // distance du l-ieme voisin, dans la direction d
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
    kdTree_sl->annkSearch(            // search
                       dataPts_sl[i], //queryPt,            // query point
                       k+ANN_ALLOW_SELF_MATCH,  // number of near neighbors (including or excluding central point)
                       nnIdx_sl[core],                // nearest neighbors (returned)
                       dists_sl[core],                // distance (returned)
                       ANN_eps,
                       core);
// 2021-12-13, test   
#ifdef DEBUG 
    if (i==DEBUG_N)
    {   std::cout << "\t[ANN_find_distance_in] with k=" << k+ANN_ALLOW_SELF_MATCH << "\n";
        std::cout << "\tnnIdx = [ ";
        for (int q=0; q<k+ANN_ALLOW_SELF_MATCH; q++) std::cout << nnIdx_sl[core][q] << " ";
        std::cout << "]\n\tdists = [ ";
        for (int q=0; q<k+ANN_ALLOW_SELF_MATCH; q++) std::cout << dists_sl[core][q] << " ";
        std::cout << "]\n";
    }
#endif    
    return( ((nnIdx_sl[core][k-1+ANN_ALLOW_SELF_MATCH]<0) ? 0. : (double)dists_sl[core][k-1+ANN_ALLOW_SELF_MATCH]) ); 
    // distance from central point
} /* end of function "ANN_find_distance_in" ****************************************/



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
    
    kdTree_sl->annkSearch(x, // queryPt,            // query point
                       k+ANN_ALLOW_SELF_MATCH,  // number of near neighbors (including or excluding central point)
                       nnIdx_sl[core],                // nearest neighbors (returned)
                       dists_sl[core],                // distance (returned)
                       ANN_eps);

    return((double)dists_sl[core][k-1+ANN_ALLOW_SELF_MATCH]); // distance from central point
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
int ANN_count_nearest_neighbors_nd_tree1_sl(double *x0, double epsilon, int core)
{   return(kdTree_1_sl->annkFRSearch(x0,   // query point
                        epsilon,        // squared radius (same as radius for L^\infty norm)
                                        // algo 1 has condition "<epsilon", hence it requires a factor
                                        // ( correction = (1.-1./ANNnpts) from calling function)
                        0,              // (number of near neighbors to return), k=0 to search and count
                        NULL, // nnIdx_1[core],  // nearest neighbors (returned if !=NULL)
                        NULL, // dists_1[core],  // distance (returned if !=NULL)
                        ANN_eps,
                        core)); 
}
int ANN_count_nearest_neighbors_nd_tree2_sl(double *x0, double epsilon, int core)
{   return(kdTree_2_sl->annkFRSearch(x0,   // query point
                        epsilon,        // squared radius (same as radius for L^\infty norm)
                                        // algo 1 has condition "<epsilon", hence it requires a factor
                                        // ( correction = (1.-1./ANNnpts) from calling function)
                        0,              // (number of near neighbors to return), k=0 to search and count
                        NULL, // nnIdx_2[core],  // nearest neighbors (returned if !=NULL)
                        NULL, // dists_2[core],  // distance (returned if !=NULL)
                        ANN_eps,
                        core));
}
int ANN_count_nearest_neighbors_nd_tree3_sl(double *x0, double epsilon, int core)
{   return(kdTree_3_sl->annkFRSearch(x0,   // query point
                        epsilon,        // squared radius (same as radius for L^\infty norm)
                                        // algo 1 has condition "<epsilon", hence it requires a factor
                                        // ( correction = (1.-1./ANNnpts) from calling function)
                        0,              // (number of near neighbors to return), k=0 to search and count
                        NULL, // nnIdx_3[core],  // nearest neighbors (returned if !=NULL)
                        NULL, // dists_3[core],  // distance (returned if !=NULL)
                        ANN_eps,
                        core));
}


// free_function (housekeeping) :
void free_ANN_sl(int NCORES)
{       
    for (int i=0; i<NCORES; i++) { delete nnIdx_sl[i]; }   delete [] nnIdx_sl;
    for (int i=0; i<NCORES; i++) { delete dists_sl[i]; }   delete [] dists_sl;
    delete kdTree_sl;          // clean main tree
	annDeallocPts(dataPts_sl);    
    annClose();                // done with ANN
}

// free_function (housekeeping) :
// 2019-01-30, version with 2 subspaces
void free_ANN_MI(int NCORES)
{
    delete kdTree_1_sl;
    annDeallocPts(dataPts_1_sl);

    delete kdTree_2_sl;
    annDeallocPts(dataPts_2_sl);
    
    // clean main tree and other global stuff, and then close ANN
    free_ANN_sl(2*NCORES);        // 2021-12-06 corrected  
}

// free_function (housekeeping) :
// 2019-01-31, version with 3 subspaces
// 2022-02-07, possible memory leak corrected
void free_ANN_PMI_sl(int NCORES)
{
    delete kdTree_3_sl;
    annDeallocPts(dataPts_3_sl);
    
	delete kdTree_2_sl;
    annDeallocPts(dataPts_2_sl);
    
	delete kdTree_1_sl;
    annDeallocPts(dataPts_1_sl);
    
    free_ANN_sl(3*NCORES);   // cleaning main tree and other global stuff, and closing ANN
}
