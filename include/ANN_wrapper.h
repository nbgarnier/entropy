//----------------------------------------------------------------------
// File:			ANN_wrapper.h
//		
// Description:	header file for ANN_wrapper.c, to use ANN in C
// This is a C header, and ANN_wrapper.cpp is a cpp file
//
// 2021-12-01 : added parameters "core" and "nb_cores" for multithreading
//----------------------------------------------------------------------
#ifndef ANN_WRAPPER_H
#define ANN_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif
    int init_ANN    (int maxPts, int dim,   int k, int nb_cores);
    int init_ANN_RE (int maxPts_x, int maxPts_y, int dim, int max_k, int nb_coores); // new 2025-12-13
    int init_ANN_MI (int maxPts, int dim_1, int dim_2, int max_k, int nb_cores); // new 2019
    int init_ANN_PMI(int maxPts, int dim_1, int dim_2, int dim_3, int max_k, int nb_cores);

// next line for debug only:
    void print_idx(int nb_cores, int max_k);

    int set_ANN_state(char CHOICE);     // new 2014
    int get_ANN_state(void);            // new 2014
    
    int create_kd_tree      (double *x, int npts, int n);
    int create_kd_tree_1    (double *x, int npts, int dim_1);
    int create_kd_tree_2    (double *x, int npts, int dim_2);
    int create_kd_tree_3    (double *x, int npts, int dim_3);
    
//    int search_ANN_internal(int i,          int n, int k, int *li, double *epsilon_z);
//    int search_ANN_external(double *x,      int n, int k, int *li, double *epsilon_z); // new 2017
    int ANN_marginal_distances_ex       (double *x, int n, int k, double *epsilon_z, int core); // new 2019, 2021-12-01
    double ANN_find_distance_in         (int i,     int n, int k, int core); // new 2019, 2021-12-01
    double ANN_find_distance_in_tree1   (int i,     int n, int k, int core); // new 2025-02-13
    double ANN_find_distance_ex         (double *x, int n, int k, int core); // new 2019
    
    int ANN_count_nearest_neighbors_nd_tree1(double *x0, double epsilon, int core); // new 2019
    int ANN_count_nearest_neighbors_nd_tree2(double *x0, double epsilon, int core); // new 2019
    int ANN_count_nearest_neighbors_nd_tree3(double *x0, double epsilon, int core); // new 2019
    
    void free_ANN    (int nb_cores);
    void free_ANN_RE (int nb_cores);
    void free_ANN_MI (int nb_cores);
    void free_ANN_PMI(int nb_cores);
#ifdef __cplusplus
}
#endif


#endif
