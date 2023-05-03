//----------------------------------------------------------------------
// File:			ANN_wrapper_second_level.h
//		
// Description:	header file for ANN_wrapper_second_level.cpp
//
// 2021-12-01 : added parameters "core" and "nb_cores" for multithreading
// 2022-11-30 : forked from "nb_cores" from "ANN_wrapper.cpp"
//----------------------------------------------------------------------
#ifndef ANN_WRAPPER_SECOND_LEVEL_H
#define ANN_WRAPPER_SECOND_LEVEL_H

#ifdef __cplusplus
extern "C" {
#endif
    int init_ANN_sl    (int maxPts, int dim,   int k, int nb_cores);
    int init_ANN_MI_sl (int maxPts, int dim_1, int dim_2, int max_k, int nb_cores); // new 2019
    int init_ANN_PMI_sl(int maxPts, int dim_1, int dim_2, int dim_3, int max_k, int nb_cores);

// next line for debug only:
    void print_idx_sl  (int nb_cores, int max_k);

    int create_kd_tree_sl   (double *x, int npts, int n);
    int create_kd_tree_1_sl (double *x, int npts, int dim_1);
    int create_kd_tree_2_sl (double *x, int npts, int dim_2);
    int create_kd_tree_3_sl (double *x, int npts, int dim_3);
    
    int ANN_marginal_distances_ex_sl (double *x, int n, int k, double *epsilon_z, int core); // new 2019, 2021-12-01
    double ANN_find_distance_in_sl   (int i,     int n, int k, int core); // new 2019, 2021-12-01
    double ANN_find_distance_ex_sl   (double *x, int n, int k, int core); // new 2019
    
    int ANN_count_nearest_neighbors_nd_tree1_sl(double *x0, double epsilon, int core); // new 2019
    int ANN_count_nearest_neighbors_nd_tree2_sl(double *x0, double epsilon, int core); // new 2019
    int ANN_count_nearest_neighbors_nd_tree3_sl(double *x0, double epsilon, int core); // new 2019
    
    void free_ANN_sl    (int nb_cores);
    void free_ANN_MI_sl (int nb_cores);
    void free_ANN_PMI_sl(int nb_cores);
#ifdef __cplusplus
}
#endif


#endif
