/*
 *  nns_count.c
 *
 *  Created by Nicolas Garnier on 2019/01/23.
 *  Copyright 2010-2019 ENS-Lyon CNRS. All rights reserved.
 *
 * 2019-01-23 : fork from entropy_nns.c
 *
 */

#include <math.h>	 /* for fabs */
#include <string.h>
#include <gsl/gsl_sf.h>  /* for psi digamma function */ 
#include "math_tools.h"


#define DEBUG		// for debug information, replace "noDEBUG" by "DEBUG"
#define LOOK 167	// for debug also (of which point(s) will we save the data ?)



/***************************************************************************************
 * count_nearest_neighbors_1d_algo1													
 * to count nearest neighbors of a point y0 in a ball of radius epsilon in 1-dimension	
 *																			
 * points on the boundary of the ball are not counted (algo 1)				
 *																	
 * y contains data, ordered along itself									
 * nx is the size of the data											
 * i is the index in the referential of the ordered data y_tmp		
 * 2011-02-16 : little change								
 * 2011-04-12 : function's name changed (more explicit)	
 * 2011-04-18 : replaced (im>=0) by (im>0) in first while loop condition
 *              replaced (ip<=) by (ip<) in second while loop condition
 * 2019-01-23 : as of today, this function is not used anymore
 *              it is kept here for reference only
 ***************************************************************************************/
int count_nearest_neighbors_1d_algo1(double *y, int nx, int i, double y0, double epsilon)
{	int im, ip;
		
	im = i; // 2011-04-18 : replaced im>=0 by im>0 below : 
	while ( (im>0) && ( (y0-y[im])<epsilon ) )			im--;
	if  ( (y0-y[im])>=epsilon )							im++; 
		
	ip = i; // 2011-04-18 : replaced ip<= by ip< below :
	while ( (ip<(nx-1)) && ( (y[ip]-y0)<epsilon ) )		ip++; 
	if ( (y[ip]-y0) >= epsilon )						ip--;
	
	return(ip-im); // on ne compte pas le point central 
} /* end of function count_nearest_neighbors_1d_algo1 **********************************/



/**************************************************************************************
 * count_nearest_neighbors_1d_algo2
 * to count nearest neighbors of a point y0 in a ball of radius epsilon in 1-dimension
 *												
 * points on the boundary of the ball are counted (algo 2)			
 *															
 * y contains data, ordered along itself				
 * nx is the size of the data													
 * i is the index in the referential of the ordered data y_tmp						
 * 2011-02-16 : little change													
 * 2011-04-12 : function's name changed (more explicit)
 * 2011-04-18 : replaced (im>=0) by (im>0) in first while loop condition
 *              replaced (ip<=) by (ip<) in second while loop condition
 * 2019-01-23 : as of today, this function is not used anymore
 *              it is kept here for reference only
 **************************************************************************************/
int count_nearest_neighbors_1d_algo2(double *y, int nx, int i, double y0, double epsilon)
{	int im, ip;
		
	im = i; // 2011-04-18 : replaced im>=0 by im>0 below :
	while ( (im>0) && ( (y0-y[im]) <= epsilon ) )		im--; 
	if  ( (y0-y[im]) > epsilon )						im++; 
	
	ip = i; // 2011-04-18 : replaced ip<= by ip< below :
	while ( (ip<(nx-1)) && ( (y[ip]-y0) <= epsilon ) )	ip++; 
	if (  (y[ip]-y0) > epsilon )						ip--;
	
	return(ip-im); // on compte le point central si on rajoute +1
} /* end of function count_nearest_neighbors_1d_algo2 *********************************/





/**************************************************************************************
 * count_nearest_neighbors_nd_algo1							
 * to count nearest neighbors of a point y0 in a ball of radius epsilon in n-dimensions
 *
 * y  contains data, ordered along itself in the first direction only	
 * nx is the size of the data in time			
 * n  is the dimension of the data			
 * i  is the index in the referential of the ordered data y_tmp	
 * ind_inv contains the permutations to recover original positionning of the data
 *							
 * 2010-11-02 : first version (working in 1-d only !)	
 * 2010-11-04 : using necessary permutations for supplementary d>0 dimensions 
 *			  but not tested	
 * 2011-07-14 : modified algorithm, using permutations in directions 0 only
 *************************************************************************************/
int count_nearest_neighbors_nd_algo1(double *y, int nx, int n, int i, int *ind_inv, double *y0, double epsilon)
{	register int d;
	int im, ip;
	int n_good, good;
	
	im     = i; 
	n_good = 0;
	while ( (im>=0) && ( (y0[0]-y[im])<epsilon ) )
	{	// est-ce bon dans les autres directions ?
		good = 1;
		for (d=1; d<n; d++)
		{	// if (fabs(y0[d]-y[d*nx+(ind_inv+d*nx)[im]])>=epsilon) good=0; // pas bon dans la direction d
			// 2011-07-14 : replaced line above, by line below :
			if (fabs(y0[d]-y[d*nx +(ind_inv+0*nx)[im]])>=epsilon)
            {   good=0; // pas bon dans la direction d
                break;    // 2019-01-26: this is a way to stop the loop earlier and gain some computation time
            }
		}
		n_good += good;
		
		if (im>=0) im--; 
	}
	im++; /* nous avons compté un point de trop */
	
	ip = i;
	while ( (ip<=(nx-1)) && ( (y[ip]-y0[0])<epsilon ) )
	{	// est-on bon dans les autres directions ?
		good = 1;
		for (d=1; d<n; d++)
		{	// if (fabs(y0[d]-y[d*nx+(ind_inv+d*nx)[ip]])>=epsilon) good=0; // pas bon dans la direction d
			// 2011-07-14 : replaced line above, by line below :
			if (fabs(y0[d]-y[d*nx+(ind_inv+0*nx)[ip]])>=epsilon)
            {   good=0; // pas bon dans la direction d
                break;
            }
		}
		n_good += good;
		
		if (ip<=(nx-1)) ip++; 
	}
	ip--;
	
	n_good -= 2;

#ifdef DEBUG	
	if (n==1)
	{	if ((ip-im)!=n_good) printf("[count_nearest_neighbors_nd_algo1] i=%d ip = %d  im = %d  ip-im = %d != found = %d\n", i, ip, im, ip-im, n_good);
	}
	else
	{	if ((ip-im)<n_good)  printf("[count_nearest_neighbors_nd_algo1] i=%d ip = %d  im = %d  ip-im = %d < found = %d\n", i, ip, im, ip-im, n_good);
	}
#endif	
	
	return(n_good);
} /* end of function count_nearest_neighbors_nd_algo1 **********************************/



/***************************************************************************************
 * idem count_nearest_neighbors_nd_algo1, but for algo 2
 * (using <= eps instead of < eps)					
 ***************************************************************************************/
int count_nearest_neighbors_nd_algo2(double *y, int nx, int n, int i, int *ind_inv, double *y0, double epsilon)
{	register int d;
	int im, ip;
	int n_good, good;
	
	im     = i;
	n_good = 0;
	while ( (im>=0) && ( (y0[0]-y[im])<=epsilon ) )
	{	// est-on bon dans les autres directions ?
		good = 1;
		for (d=1; d<n; d++)
		{	if (fabs(y0[d]-y[d*nx+(ind_inv+0*nx)[im]])>epsilon)
            {   good=0; // pas bon dans la direction d
                break;
            }
		}
		n_good += good;
		
		if (im>=0) im--; 
	}
	im++; /* nous avons compté un point de trop */
	
	ip = i;
	while ( (ip<=(nx-1)) && ( (y[ip]-y0[0])<=epsilon ) )
	{	// est-on bon dans les autres directions ?
		good = 1;
		for (d=1; d<n; d++)
		{	if (fabs(y0[d]-y[d*nx+(ind_inv+0*nx)[ip]])>epsilon)
            {   good=0; // pas bon dans la direction d
                break;
            }
	}
		n_good += good;
		
		if (ip<=(nx-1)) ip++; 
	}
	ip--;
	
	n_good -= 2;

#ifdef DEBUG	
	if (n==1)
	{	if ((ip-im)!=n_good) printf("[count_nearest_neighbors_nd_algo2] i=%d ip = %d  im = %d  ip-im = %d != found = %d\n", i, ip, im, ip-im, n_good);
	}
	else
	{	if ((ip-im)<n_good)  printf("[count_nearest_neighbors_nd_algo2] i=%d ip = %d  im = %d  ip-im = %d < found = %d\n", i, ip, im, ip-im, n_good);
	}
#endif
	
	return(n_good);
} /* end of function count_nearest_neighbors_nd_algo2 **********************************/




/***************************************************************************************
 * idem as "count_nearest_neighbors_nd_algo2", but slightly better, as it takes into
 * account a different epsilon in each direction, which is required by algorithm 2
 *							
 * 2010-11-07 : first version	
 * 2011-07-14 : modified according to "count_nearest_neighbors_nd_algo1"
 ***************************************************************************************/
int count_nearest_neighbors_nd_algo2n(double *y, int nx, int n, int i, int *ind_inv, double *y0, double *epsilon)
{	register int d;
	int im, ip;
	int n_good, good;
	
	im     = i;
	n_good = 0;
	while ( (im>=0) && ( (y0[0]-y[im])<=epsilon[0] ) )
	{	// est-on bon dans les autres directions ?
		good = 1;
		for (d=1; d<n; d++)
		{	if (fabs(y0[d]-y[d*nx+(ind_inv+0*nx)[im]])>epsilon[d])
            {   good=0; // pas bon dans la direction d
                break;
            }
		}
		n_good += good;
		
		if (im>=0) im--; 
	}
	im++; /* nous avons compté un point de trop */
	
	ip = i; // i0;
	while ( (ip<=(nx-1)) && ( (y[ip]-y0[0])<=epsilon[0] ) )
	{	// est-on bon dans les autres directions ?
		good = 1;
		for (d=1; d<n; d++)
		{	if (fabs(y0[d]-y[d*nx+(ind_inv+0*nx)[ip]])>epsilon[d])
            {   good=0; // pas bon dans la direction d
                break;
            }
		}
		n_good += good;
		
		if (ip<=(nx-1)) ip++; 
	}
	ip--;
	
	n_good -= 2;

#ifdef DEBUG	
	if (n==1)
	{	if ((ip-im)!=n_good) printf("[count_nearest_neighbors_nd_algo2n] i=%d ip = %d  im = %d  ip-im = %d != found = %d\n", i, ip, im, ip-im, n_good);
	}
	else
	{	if ((ip-im)<n_good)  printf("[count_nearest_neighbors_nd_algo2n] i=%d ip = %d  im = %d  ip-im = %d < found = %d\n", i, ip, im, ip-im, n_good);
	}
#endif

	return(n_good);
} /* end of function count_nearest_neighbors_nd_algo2n *********************************/


