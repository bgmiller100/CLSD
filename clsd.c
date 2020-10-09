/*----------------------------------------------------------------------------
CLSD - Conditional line segment detection, using detections on prior noise models to inform observational detections, assuming non-uniformity of background noise.

This code is with the publication below:

[TO BE ADDED]

*NOTICE: This program is modified from the source code of LSDSAR:
*"LSDSAR, a Markovian a contrario framework for line segment detection in SAR images"
*by Chenguang Liu, Rémy Abergel, Yann Gousseau and Florence Tupin. 
*(Pattern Recognition, 2019).
*https://doi.org/10.1016/j.patcog.2019.107034
*Date of Modification: October 8, 2020.

*NOTICE: This program is released under [INSERT LICENSING STATEMENT FROM GIT]

--------------------------------------------------------------------------------------------


*NOTICE: This program is modified from the source code of LSDSAR:
*"LSDSAR, a Markovian a contrario framework for line segment detection in SAR images"
*by Chenguang Liu, Rémy Abergel, Yann Gousseau and Florence Tupin. 
*(Pattern Recognition, 2019).
*https://doi.org/10.1016/j.patcog.2019.107034
*Date of Modification: October 8, 2020.

Below, unmarked numbers indicate significantly new functions, otherwise:
	*   Edited from 'double * LineSegmentDetection'
	**  Edited from similarly-named 'double * lsd', 'lsd_scale', 'lsd_scale_region'
	*** Edited from 'static void add-7tuple'

Major modifications are:

	1)  double * CLSD_Pipeline
	2*) double * Conditinal_LineSegmentDetection
   	3*) static void make_markov
	4)  double * surf_grid
	5)  double * sort_list
		
Minor modifications are: 

	6**)  double * c_lsd
	7**)  double * c_lsd_scale
	8**)  double * c_lsd_scale_region
	9***) static void add_4tuple 	
	10)   struct sortstr
	11)   int cmp
	12)   static double line_angle
	13)   double calc_mean 
	14)   double calc_std 

New Python wrapper functions are:

	15) PyObject * clsd( PyObject * self, PyObject * args)
	16) static PyMethodDef clsdMethods[]
	17) static struct PyModuleDef moduledef
	18) PyMODINIT_FUNC PyInit_clsd(void)

The Python interface requires building the program using 'setup.py'
to link against the 'gsl' and 'mir' libraries for conditional interpolation.
Building has been confirmed for Python 3.7.6 on Linux 5.4.0-48

The other functions of the code are kept unchanged.

Benjamin Miller
The University of Texas at Austin
Email: benjamin.g.miller@utexas.edu

*/
/*
*****************************************************************************
*****************************************************************************
**Here is the header file of the original LSDSAR.
**-------------------------------------------------------------------------------------------------------
**----------------------------------------------------------------------------
LSDSAR-line segment detector for SAR images.

This code is with the publication below:

 "LSDSAR, a Markovian a contrario framework for line segment detection in SAR images",
 by Chenguang Liu, Rémy Abergel, Yann Gousseau and Florence Tupin. (Pattern Recognition, 2019).

*NOTICE: This program is modified from the source code of LSD:
*"LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
*Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
*Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
*http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
*Date of Modification: 27/06/2018.

*NOTICE: This program is released under GNU Affero General Public License
*and any conditions added under section 7 in the link:
*https://www.gnu.org/licenses/agpl-3.0.en.html

Copyright (c) 2017, 2018 Chenguang Liu <chenguangl@whu.edu.cn>

This program is free software: you can redistribute it and/or modify 
 it under the terms of the GNU General Public License as published 
 by the Free Software Foundation, either version 3 of the License, 
 or (at your option) any later version.

This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.
 
You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------------------

*NOTICE: This code is modified from the source code of LSD:
*"LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
*Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
*Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
*http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
*Date of Modification: 27/06/2018.

The modifications lie in functions:
    1) double * lsd(int * n_out, double * img, int X, int Y),
    2) double * lsd_scale(int * n_out, double * img, int X, int Y, double scale),
    3) double * lsd_scale_region( int * n_out,
                           double * img, int X, int Y, double scale,
                           int ** reg_img, int * reg_x, int * reg_y ),
    4)double * LineSegmentDetection( int * n_out,
                               double * img, int X, int Y,
                               double scale, double sigma_scale, double quant,
                               double ang_th, double log_eps, double density_th,
                               int n_bins,
                               int ** reg_img, int * reg_x, int * reg_y ),
    5) static image_double ll_angle( image_double in, double threshold,
                              struct coorlist ** list_p, void ** mem_p,
                              image_double * modgrad, unsigned int n_bins ),
    6) static int refine( struct point * reg, int * reg_size, image_double modgrad,
                   double reg_angle, double prec, double p, struct rect * rec,
                   image_char used, image_double angles, double density_th ),
    7) static int reduce_region_radius( struct point * reg, int * reg_size,
                                 image_double modgrad, double reg_angle,
                                 double prec, double p, struct rect * rec,
                                 image_char used, image_double angles,
                                 double density_th ),
    8) static double rect_improve( struct rect * rec, image_double angles,
                            double logNT, double log_eps ),
    9) static double rect_nfa(struct rect * rec, image_double angles, double logNT),
    10) static double nfa(int n, int k, double p, double logNT).

The other functions of the code are kept unchanged.

 I would be grateful to receive any advices or possible erros in the source code. 

Chenguang Liu
Telecom ParisTech
Email: chenguang.liu@telecom-paristech.fr
Email: chenguangl@whu.edu.cn (permanent)
*/
/*
*****************************************************************************
*****************************************************************************
**Here is the header file of the original LSD.
**-------------------------------------------------------------------------------------------------------
**-------------------------------------------------------------------------------------------------------
**     
**   LSD - Line Segment Detector on digital images
**
**  This code is part of the following publication and was subject
**  to peer review:
**
**   "LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
**    Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
**    Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
**    http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
**
**  Copyright (c) 2007-2011 rafael grompone von gioi <grompone@gmail.com>
**
**  This program is free software: you can redistribute it and/or modify
**  it under the terms of the GNU Affero General Public License as
**  published by the Free Software Foundation, either version 3 of the
**  License, or (at your option) any later version.
**
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
**  GNU Affero General Public License for more details.
**
**  You should have received a copy of the GNU Affero General Public License
**  along with this program. If not, see <http://www.gnu.org/licenses/>.
**
**  ----------------------------------------------------------------------------*/

//Headers for LSD
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include<string.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_qrng.h>
#include<mir/mir.h>
#include "clsd.h"

//Header for interpolant



/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif /* !M_LN10 */

/** PI */
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */

#define RADIANS_TO_DEGREES (180.0/M_PI)
#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/** Label for pixels with undefined gradient. */
#define NOTDEF -1024.0

/** 3/2 pi */
#define M_3_2_PI 4.71238898038

/** 2 pi */
#define M_2__PI  6.28318530718

/** Label for pixels not used in yet. */
#define NOTUSED 0

/** Label for pixels already used in detection. */
#define USED    1

/*----------------------------------------------------------------------------*/
/** Chained list of coordinates.
 */
int max1(int x, int y)
{if(x<y)
return y;
 else
return x;}
int min1(int x, int y)
{return x<y?x:y;}
struct coorlist
{
  int x,y;
  struct coorlist * next;
};

/*----------------------------------------------------------------------------*/
/** A point (or pixel).
 */
struct point {int x,y;};


/*----------------------------------------------------------------------------*/
/*------------------------- Miscellaneous functions --------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Fatal error, print a message to standard-error output and exit.
 */
static void error(char * msg)
{
  fprintf(stderr,"LSD Error: %s\n",msg);
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*/
/** Doubles relative error factor
 */
#define RELATIVE_ERROR_FACTOR 100.0

/*----------------------------------------------------------------------------*/
/** Compare doubles by relative error.

    The resulting rounding error after floating point computations
    depend on the specific operations done. The same number computed by
    different algorithms could present different rounding errors. For a
    useful comparison, an estimation of the relative rounding error
    should be considered and compared to a factor times EPS. The factor
    should be related to the cumulated rounding error in the chain of
    computation. Here, as a simplification, a fixed factor is used.
 */
static int double_equal(double a, double b)
{
  double abs_diff,aa,bb,abs_max;

  /* trivial case */
  if( a == b ) return TRUE;

  abs_diff = fabs(a-b);
  aa = fabs(a);
  bb = fabs(b);
  abs_max = aa > bb ? aa : bb;

  /* DBL_MIN is the smallest normalized number, thus, the smallest
     number whose relative error is bounded by DBL_EPSILON. For
     smaller numbers, the same quantization steps as for DBL_MIN
     are used. Then, for smaller numbers, a meaningful "relative"
     error should be computed by dividing the difference by DBL_MIN. */
  if( abs_max < DBL_MIN ) abs_max = DBL_MIN;

  /* equal if relative error <= factor x eps */
  return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
}

/*----------------------------------------------------------------------------*/
/** Computes Euclidean distance between point (x1,y1) and point (x2,y2).
 */
static double dist(double x1, double y1, double x2, double y2)
{
  return sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) );
}

/*----------------------------------------------------------------------------*/
/* Orientation of a line*/
static double line_angle(double x1, double y1, double x2, double y2)
{
  return atan2( y2-y1 , x2-x1 );
}


/*----------------------------------------------------------------------------*/
/* Parameter-wise mean calculation for  ndim-by-nv vector of observationse*/
double calc_mean(double* xv, int nv, int ndim,int param)
{
	double mean=0;
	int i;
	for(i=0;i<nv;i++){mean+=xv[i*ndim+param];}
	mean/=nv;
	return mean;
}

/*----------------------------------------------------------------------------*/
/* Parameter-wise deviation calculation for  ndim-by-nv vector of observationse*/
double calc_std(double* xv, int nv, int ndim,int param)
{
	double std=0;
	int i;
	double mean = calc_mean(xv,nv,ndim,param);
	for(i=0;i<nv;i++){std+=pow(xv[i*ndim+param]-mean,2);}
	return sqrt(std/nv);
}
/*----------------------------------------------------------------------------*/
/*----------------------- 'list of n-tuple' data type ------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** 'list of n-tuple' data type

    The i-th component of the j-th n-tuple of an n-tuple list 'ntl'
    is accessed with:

      ntl->values[ i + j * ntl->dim ]

    The dimension of the n-tuple (n) is:

      ntl->dim

    The number of n-tuples in the list is:

      ntl->size

    The maximum number of n-tuples that can be stored in the
    list with the allocated memory at a given time is given by:

      ntl->max_size
 */
typedef struct ntuple_list_s
{
  unsigned int size;
  unsigned int max_size;
  unsigned int dim;
  double * values;
} * ntuple_list;

/*----------------------------------------------------------------------------*/
/** Free memory used in n-tuple 'in'.
 */
static void free_ntuple_list(ntuple_list in)
{
  if( in == NULL || in->values == NULL )
    error("free_ntuple_list: invalid n-tuple input.");
  free( (void *) in->values );
  free( (void *) in );
}

/*----------------------------------------------------------------------------*/
/** Create an n-tuple list and allocate memory for one element.
    @param dim the dimension (n) of the n-tuple.
 */
static ntuple_list new_ntuple_list(unsigned int dim)
{
  ntuple_list n_tuple;

  /* check parameters */
  if( dim == 0 ) error("new_ntuple_list: 'dim' must be positive.");

  /* get memory for list structure */
  n_tuple = (ntuple_list) malloc( sizeof(struct ntuple_list_s) );
  if( n_tuple == NULL ) error("not enough memory.");

  /* initialize list */
  n_tuple->size = 0;
  n_tuple->max_size = 1;
  n_tuple->dim = dim;

  /* get memory for tuples */
  n_tuple->values = (double *) malloc( dim*n_tuple->max_size * sizeof(double) );
  if( n_tuple->values == NULL ) error("not enough memory.");

  return n_tuple;
}

/*----------------------------------------------------------------------------*/
/** Enlarge the allocated memory of an n-tuple list.
 */
static void enlarge_ntuple_list(ntuple_list n_tuple)
{
  /* check parameters */
  if( n_tuple == NULL || n_tuple->values == NULL || n_tuple->max_size == 0 )
    error("enlarge_ntuple_list: invalid n-tuple.");

  /* duplicate number of tuples */
  n_tuple->max_size *= 2;

  /* realloc memory */
  n_tuple->values = (double *) realloc( (void *) n_tuple->values,
                      n_tuple->dim * n_tuple->max_size * sizeof(double) );
  if( n_tuple->values == NULL ) error("not enough memory.");
}

/*----------------------------------------------------------------------------*/
/** Add a 7-tuple to an n-tuple list.
 */
static void add_7tuple( ntuple_list out, double v1, double v2, double v3,
                        double v4, double v5, double v6, double v7 )
{
  /* check parameters */
  if( out == NULL ) error("add_7tuple: invalid n-tuple input.");
  if( out->dim != 7 ) error("add_7tuple: the n-tuple must be a 7-tuple.");

  /* if needed, alloc more tuples to 'out' */
  if( out->size == out->max_size ) enlarge_ntuple_list(out);
  if( out->values == NULL ) error("add_7tuple: invalid n-tuple input.");

  /* add new 7-tuple */
  out->values[ out->size * out->dim + 0 ] = v1;
  out->values[ out->size * out->dim + 1 ] = v2;
  out->values[ out->size * out->dim + 2 ] = v3;
  out->values[ out->size * out->dim + 3 ] = v4;
  out->values[ out->size * out->dim + 4 ] = v5;
  out->values[ out->size * out->dim + 5 ] = v6;
  out->values[ out->size * out->dim + 6 ] = v7;

  /* update number of tuples counter */
  out->size++;
}


/*----------------------------------------------------------------------------*/
/** Add a 4-tuple to an n-tuple list.
 */
static void add_4tuple( ntuple_list out, double v1, double v2, double v3, double v4 )
{
  /* check parameters */
  if( out == NULL ) error("add_4tuple: invalid n-tuple input.");
  if( out->dim != 4 ) error("add_4tuple: the n-tuple must be a 7-tuple.");

  /* if needed, alloc more tuples to 'out' */
  if( out->size == out->max_size ) enlarge_ntuple_list(out);
  if( out->values == NULL ) error("add_4tuple: invalid n-tuple input.");

  /* add new 7-tuple */
  out->values[ out->size * out->dim + 0 ] = v1;
  out->values[ out->size * out->dim + 1 ] = v2;
  out->values[ out->size * out->dim + 2 ] = v3;
  out->values[ out->size * out->dim + 3 ] = v4;
  /* update number of tuples counter */
  out->size++;
}
/*----------------------------------------------------------------------------*/
/*----------------------------- Image Data Types -----------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** char image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image_char_s
{
  unsigned char * data;
  unsigned int xsize,ysize;
} * image_char;

/*----------------------------------------------------------------------------*/
/** Free memory used in image_char 'i'.
 */
static void free_image_char(image_char i)
{
  if( i == NULL || i->data == NULL )
    error("free_image_char: invalid input image.");
  free( (void *) i->data );
  free( (void *) i );
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize'.
 */
static image_char new_image_char(unsigned int xsize, unsigned int ysize)
{
  image_char image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 ) error("new_image_char: invalid image size.");

  /* get memory */
  image = (image_char) malloc( sizeof(struct image_char_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (unsigned char *) calloc( (size_t) (xsize*ysize),
                                          sizeof(unsigned char) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
static image_char new_image_char_ini( unsigned int xsize, unsigned int ysize,
                                      unsigned char fill_value )
{
  image_char image = new_image_char(xsize,ysize); /* create image */
  unsigned int N = xsize*ysize;
  unsigned int i;

  /* check parameters */
  if( image == NULL || image->data == NULL )
    error("new_image_char_ini: invalid image.");

  /* initialize */
  for(i=0; i<N; i++) image->data[i] = fill_value;

  return image;
}

/*----------------------------------------------------------------------------*/
/** int image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image_int_s
{
  int * data;
  unsigned int xsize,ysize;
} * image_int;

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize'.
 */
static image_int new_image_int(unsigned int xsize, unsigned int ysize)
{
  image_int image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 ) error("new_image_int: invalid image size.");

  /* get memory */
  image = (image_int) malloc( sizeof(struct image_int_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (int *) calloc( (size_t) (xsize*ysize), sizeof(int) );
  if( image->data == NULL ) error("not enough memory.");

  /* set imagsavelines=zeros(size(lines,1),5);
for xx=1:size(lines,1)
    savelines(xx,1:4)=lines(xx,1:4);
    savelines(xx,5)=angleline(xx);
ende size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
static image_int new_image_int_ini( unsigned int xsize, unsigned int ysize,
                                    int fill_value )
{
  image_int image = new_image_int(xsize,ysize); /* create image */
  unsigned int N = xsize*ysize;
  unsigned int i;

  /* initialize */
  for(i=0; i<N; i++) image->data[i] = fill_value;

  return image;
}

/*----------------------------------------------------------------------------*/
/** double image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image_double_s
{
  double * data;
  unsigned int xsize,ysize;
} * image_double;

/*----------------------------------------------------------------------------*/
/** Free memory used in image_double 'i'.
 */
static void free_image_double(image_double i)
{
  if( i == NULL || i->data == NULL )
    error("free_image_double: invalid input image.");
  free( (void *) i->data );
  free( (void *) i );
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'.
 */
static image_double new_image_double(unsigned int xsize, unsigned int ysize)
{
  image_double image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 ) error("new_image_double: invalid image size.");

  /* get memory */
  image = (image_double) malloc( sizeof(struct image_double_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (double *) calloc( (size_t) (xsize*ysize), sizeof(double) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'
    with the data pointed by 'data'.
 */
static image_double new_image_double_ptr( unsigned int xsize,
                                          unsigned int ysize, double * data )
{
  image_double image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 )
    error("new_image_double_ptr: invalid image size.");
  if( data == NULL ) error("new_image_double_ptr: NULL data pointer.");

  /* get memory */
  image = (image_double) malloc( sizeof(struct image_double_s) );
  if( image == NULL ) error("not enough memory.");

  /* set image */
  image->xsize = xsize;
  image->ysize = ysize;
  
  image->data = data;

  return image;
}


/*----------------------------------------------------------------------------*/
/*----------------------------- Gaussian filter ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Compute a Gaussian kernel of length 'kernel->dim',
    standard deviation 'sigma', and centered at value 'mean'.

    For example, if mean=0.5, the Gaussian will be centered
    in the middle point between values 'kernel->values[0]'
    and 'kernel->values[1]'.
 */
static void gaussian_kernel(ntuple_list kernel, double sigma, double mean)
{
  double sum = 0.0;
  double val;
  unsigned int i;

  /* check parameters */
  if( kernel == NULL || kernel->values == NULL )
    error("gaussian_kernel: invalid n-tuple 'kernel'.");
  if( sigma <= 0.0 ) error("gaussian_kernel: 'sigma' must be positive.");

  /* compute Gaussian kernel */
  if( kernel->max_size < 1 ) enlarge_ntuple_list(kernel);
  kernel->size = 1;
  for(i=0;i<kernel->dim;i++)
    {
      val = ( (double) i - mean ) / sigma;
      kernel->values[i] = exp( -0.5 * val * val );
      sum += kernel->values[i];
    }

  /* normalization */
  if( sum >= 0.0 ) for(i=0;i<kernel->dim;i++) kernel->values[i] /= sum;
}

/*----------------------------------------------------------------------------*/
/** Scale the input image 'in' by a factor 'scale' by Gaussian sub-sampling.

    For example, scale=0.8 will give a result at 80% of the original size.

    The image is convolved with a Gaussian kernel
    @f[
        G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}
    @f]
    before the sub-sampling to prevent aliasing.

    The standard deviation sigma given by:
    -  sigma = sigma_scale / scale,   if scale <  1.0
    -  sigma = sigma_scale,           if scale >= 1.0

    To be able to sub-sample at non-integer steps, some interpolation
    is needed. In this implementation, the interpolation is done by
    the Gaussian kernel, so both operations (filtering and sampling)
    are done at the same time. The Gaussian kernel is computed
    centered on the coordinates of the required sample. In this way,
    when applied, it gives directly the result of convolving the image
    with the kernel and interpolated to that particular position.

    A fast algorithm is done using the separability of the Gaussian
    kernel. Applying the 2D Gaussian kernel is equivalent to applying
    first a horizontal 1D Gaussian kernel and then a vertical 1D
    Gaussian kernel (or the other way round). The reason is that
    @f[
        G(x,y) = G(x) * G(y)
    @f]
    where 
    @f[
        G(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}.
    @f]
    The algorithm first applies a combined Gaussian kernel and sampling
    in the x axis, and then the combined Gaussian kernel and sampling
    in the y axis.
 */
static image_double gaussian_sampler( image_double in, double scale,
                                      double sigma_scale )
{
  image_double aux,out;
  ntuple_list kernel;
  unsigned int N,M,h,n,x,y,i;
  int xc,yc,j,double_x_size,double_y_size;
  double sigma,xx,yy,sum,prec;

  /* check parameters */
  if( in == NULL || in->data == NULL || in->xsize == 0 || in->ysize == 0 )
    error("gaussian_sampler: invalid image.");
  if( scale <= 0.0 ) error("gaussian_sampler: 'scale' must be positive.");
  if( sigma_scale <= 0.0 )
    error("gaussian_sampler: 'sigma_scale' must be positive.");

  /* compute new image size and get memory for images */
  if( in->xsize * scale > (double) UINT_MAX ||
      in->ysize * scale > (double) UINT_MAX )
    error("gaussian_sampler: the output image size exceeds the handled size.");
  N = (unsigned int) ceil( in->xsize * scale );
  M = (unsigned int) ceil( in->ysize * scale );
  aux = new_image_double(N,in->ysize);
  out = new_image_double(N,M);

  /* sigma, kernel size and memory for the kernel */
  sigma = scale < 1.0 ? sigma_scale / scale : sigma_scale;
  /*
     The size of the kernel is selected to guarantee that the
     the first discarded term is at least 10^prec times smaller
     than the central value. For that, h should be larger than x, with
       e^(-x^2/2sigma^2) = 1/10^prec.
     Then,
       x = sigma * sqrt( 2 * prec * ln(10) ).
   */
  prec = 3.0;
  h = (unsigned int) ceil( sigma * sqrt( 2.0 * prec * log(10.0) ) );
  n = 1+2*h; /* kernel size */
  kernel = new_ntuple_list(n);

  /* auxiliary double image size variables */
  double_x_size = (int) (2 * in->xsize);
  double_y_size = (int) (2 * in->ysize);

  /* First subsampling: x axis */
  for(x=0;x<aux->xsize;x++)
    {
      /*
         x   is the coordinate in the new image.
         xx  is the corresponding x-value in the original size image.
         xc  is the integer value, the pixel coordinate of xx.
       */
      xx = (double) x / scale;
      /* coordinate (0.0,0.0) is in the center of pixel (0,0),
         so the pixel with xc=0 get the values of xx from -0.5 to 0.5 */
      xc = (int) floor( xx + 0.5 );
      gaussian_kernel( kernel, sigma, (double) h + xx - (double) xc );
      /* the kernel must be computed for each x because the fine
         offset xx-xc is different in each case */

      for(y=0;y<aux->ysize;y++)
        {
          sum = 0.0;
          for(i=0;i<kernel->dim;i++)
            {
              j = xc - h + i;

              /* symmetry boundary condition */
              while( j < 0 ) j += double_x_size;
              while( j >= double_x_size ) j -= double_x_size;
              if( j >= (int) in->xsize ) j = double_x_size-1-j;

              sum += in->data[ j + y * in->xsize ] * kernel->values[i];
            }
          aux->data[ x + y * aux->xsize ] = sum;
        }
    }

  /* Second subsampling: y axis */
  for(y=0;y<out->ysize;y++)
    {
      /*
         y   is the coordinate in the new image.
         yy  is the corresponding x-value in the original size image.
         yc  is the integer value, the pixel coordinate of xx.
       */
      yy = (double) y / scale;
      /* coordinate (0.0,0.0) is in the center of pixel (0,0),
         so the pixel with yc=0 get the values of yy from -0.5 to 0.5 */
      yc = (int) floor( yy + 0.5 );
      gaussian_kernel( kernel, sigma, (double) h + yy - (double) yc );
      /* the kernel must be computed for each y because the fine
         offset yy-yc is different in each case */

      for(x=0;x<out->xsize;x++)
        {
          sum = 0.0;
          for(i=0;i<kernel->dim;i++)
            {
              j = yc - h + i;

              /* symmetry boundary condition */
              while( j < 0 ) j += double_y_size;
              while( j >= double_y_size ) j -= double_y_size;
              if( j >= (int) in->ysize ) j = double_y_size-1-j;

              sum += aux->data[ x + j * aux->xsize ] * kernel->values[i];
            }
          out->data[ x + y * out->xsize ] = sum;
        }
    }

  /* free memory */
  free_ntuple_list(kernel);
  free_image_double(aux);

  return out;
}


/*----------------------------------------------------------------------------*/
/*--------------------------------- Gradient ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the direction of the level line of 'in' at each point.

    The result is:
    - an image_double with the angle at each pixel, or NOTDEF if not defined.
    - the image_double 'modgrad' (a pointer is passed as argument)
      with the gradient magnitude at each point.
    - a list of pixels 'list_p' roughly ordered by decreasing
      gradient magnitude. (The order is made by classifying points
      into bins by gradient magnitude. The parameters 'n_bins' and
      'max_grad' specify the number of bins and the gradient modulus
      at the highest bin. The pixels in the list would be in
      decreasing gradient magnitude, up to a precision of the size of
      the bins.)
    - a pointer 'mem_p' to the memory used by 'list_p' to be able to
      free the memory when it is not used anymore.
 */
static image_double ll_angle( image_double in,
                              struct coorlist ** list_p, void ** mem_p,
                              image_double * modgrad, unsigned int n_bins,double alpha)
{
  image_double g;
  unsigned int n,p,x,y,adr,i;
  double com1,com2,gx,gy,norm,norm2;
 
  int list_count = 0;
  struct coorlist * list;
  struct coorlist ** range_l_s; /* array of pointers to start of bin list */
  struct coorlist ** range_l_e; /* array of pointers to end of bin list */
  struct coorlist * start;
  struct coorlist * end;
  double max_grad = 0.0;

  /* check parameters */
  if( in == NULL || in->data == NULL || in->xsize == 0 || in->ysize == 0 )
    error("ll_angle: invalid image.");
 
  if( list_p == NULL ) error("ll_angle: NULL pointer 'list_p'.");
  if( mem_p == NULL ) error("ll_angle: NULL pointer 'mem_p'.");
  if( modgrad == NULL ) error("ll_angle: NULL pointer 'modgrad'.");
  if( n_bins == 0 ) error("ll_angle: 'n_bins' must be positive.");

  /* image size shortcuts */
  n = in->ysize;
  p = in->xsize;
  /* allocate output image */
  g = new_image_double(in->xsize,in->ysize);

  /* get memory for the image of gradient modulus */
  *modgrad = new_image_double(in->xsize,in->ysize);

  /* get memory for "ordered" list of pixels */
  list = (struct coorlist *) calloc( (size_t) (n*p), sizeof(struct coorlist) );
  *mem_p = (void *) list;
  range_l_s = (struct coorlist **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
  range_l_e = (struct coorlist **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
  if( list == NULL || range_l_s == NULL || range_l_e == NULL )
    error("not enough memory.");
  for(i=0;i<n_bins;i++) range_l_s[i] = range_l_e[i] = NULL;



  /* 'undefined' on the down and righ t boundaries */
  for(x=0;x<p;x++) g->data[(n-1)*p+x] = NOTDEF;
  for(y=0;y<n;y++) g->data[p*y+p-1]   = NOTDEF;

  
 int j,k;
double ax,ay,an,ap;
  int X;  /* x image size */
  int Y;  /* y image size */
Y=n;
X=p;
double beta=alpha;
int nb=16;




  /* create a simple image: left half black, right half gray */
  
  
  double * gradx;
gradx= (double *) malloc( X * Y * sizeof(double) );
double * grady;
grady= (double *) malloc( X * Y * sizeof(double) );
int imgSize=X*Y;
int bitdepth=16;
double *img1;
img1= (double *) malloc( X * Y * sizeof(double) );
double *img2;
img2= (double *) malloc( X * Y * sizeof(double) );
for(i=0;i<imgSize;i++)
{
/*img1[i]=pow(image[i],2);*/
    img1[i]=in->data[i];
if(img1[i]<1.)
img2[i]=1.;
else
img2[i]=img1[i];
}
int longueur=ceil(log(10)*beta);

/*longueur=wid;*/
int largeur=longueur;


double * gradx1;
gradx1= (double *) malloc( X * Y * sizeof(double) );
double * grady1;
grady1= (double *) malloc( X * Y * sizeof(double) );
for(j=0;j<Y;j++)
{
for(i=0;i<X;i++)
{
double Mx=0.;
double My=0.;
for(k=-largeur;k<=largeur;k++)
{
int xk=min1(max1(i+k,0),X-1);
int yk=min1(max1(j+k,0),Y-1);
double coeff=exp(-(double) abs(k)/beta);
Mx+=coeff*img2[xk+j*X];
My+=coeff*img2[i+yk*X];
}
gradx1[i+j*X]=Mx;
grady1[i+j*X]=My;

}
}
for(j=0;j<Y;j++)
{
for(i=0;i<X;i++)
{
double Mxg=0;
double Mxd=0;
double Myg=0;
double Myd=0;
for(k=1;k<=longueur;k++)
{
double coeff=exp(-(double) abs(k)/beta);
int yl1;
if(j-k<0)
    yl1=0;
else
    yl1=j-k;
int yl2;
if(j+k>Y-1)
    yl2=Y-1;
else
    yl2=j+k;
Mxg+=coeff*gradx1[i+yl1*X];
Mxd+=coeff*gradx1[i+yl2*X];
int xl1=max1(i-k,0);
int xl2=min1(i+k,X-1);;
Myg+=coeff*grady1[xl1+j*X];
Myd+=coeff*grady1[xl2+j*X];
}
gradx[i+j*X]=log(Mxd/Mxg);
grady[i+j*X]=log(Myd/Myg);
}
}
for(i=0;i<X;i++)
{
for(j=0;j<Y;j++)
{
  adr = j*X+i;
ay=gradx[adr];
ax=grady[adr];



an=(double) hypot((double) ax,(double) ay);
norm=an;


        (*modgrad)->data[adr] = norm; /* store gradient norm */

       
        if( norm <= 0.0 ) /* norm too small, gradient no defined */
          g->data[adr] = NOTDEF; /* gradient angle not defined */
        else
          {
            /* gradient angle computation */
            ap=atan2((double) ax,-(double) ay);
            g->data[adr] = ap;

            /* look for the maximum of the gradient */
            if( norm > max_grad ) max_grad = norm;
          }

         
}
}
int i0;
  /* compute histogram of gradient values */
  for(x=0;x<X-1;x++)
    for(y=0;y<Y-1;y++)
      {
        norm = (*modgrad)->data[y*p+x];

        /* store the point in the right bin according to its norm */
        i0= (unsigned int) (norm * (double) n_bins / max_grad);
        if( i0 >= n_bins ) i0 = n_bins-1;
        if( range_l_e[i0] == NULL )
          range_l_s[i0] = range_l_e[i0] = list+list_count++;
        else
          {
            range_l_e[i0]->next = list+list_count;
            range_l_e[i0] = list+list_count++;
          }
        range_l_e[i0]->x = (int) x;
        range_l_e[i0]->y = (int) y;
        range_l_e[i0]->next = NULL;
      }

  /* Make the list of pixels (almost) ordered by norm value.
     It starts by the larger bin, so the list starts by the
     pixels with the highest gradient value. Pixels would be ordered
     by norm value, up to a precision given by max_grad/n_bins.
   */
  for(i=n_bins-1; i>0 && range_l_s[i]==NULL; i--);
  start = range_l_s[i];
  end = range_l_e[i];
  if( start != NULL )
    while(i>0)
      {
        --i;
        if( range_l_s[i] != NULL )
          {
            end->next = range_l_s[i];
            end = range_l_e[i];
          }
      }
  *list_p = start;

  /* free memory */
  free( (void *) range_l_s );
  free( (void *) range_l_e );
free ((void *) gradx);
  free((void *) grady);
  free((void *) img1);
  free((void *) img2);
  free((void *) gradx1);
  free((void *) grady1);
  return g;
}

/*----------------------------------------------------------------------------*/
/** Is point (x,y) aligned to angle theta, up to precision 'prec'?
 */
static int isaligned( int x, int y, image_double angles, double theta,
                      double prec )
{
  double a;

  /* check parameters */
  if( angles == NULL || angles->data == NULL )
    error("isaligned: invalid image 'angles'.");
  if( x < 0 || y < 0 || x >= (int) angles->xsize || y >= (int) angles->ysize )
    error("isaligned: (x,y) out of the image.");
  if( prec < 0.0 ) error("isaligned: 'prec' must be positive.");

  /* angle at pixel (x,y) */
  a = angles->data[ x + y * angles->xsize ];

  /* pixels whose level-line angle is not defined
     are considered as NON-aligned */
  if( a == NOTDEF ) return FALSE;  /* there is no need to call the function
                                      'double_equal' here because there is
                                      no risk of problems related to the
                                      comparison doubles, we are only
                                      interested in the exact NOTDEF value */

  /* it is assumed that 'theta' and 'a' are in the range [-pi,pi] */
  theta -= a;
  if( theta < 0.0 ) theta = -theta;
  if( theta > M_3_2_PI )
    {
      theta -= M_2__PI;
      if( theta < 0.0 ) theta = -theta;
    }

  return theta <= prec;
}

/*----------------------------------------------------------------------------*/
/** Absolute value angle difference.
 */
static double angle_diff(double a, double b)
{
  a -= b;
  while( a <= -M_PI ) a += M_2__PI;
  while( a >   M_PI ) a -= M_2__PI;
  if( a < 0.0 ) a = -a;
  return a;
}

/*----------------------------------------------------------------------------*/
/** Signed angle difference.
 */
static double angle_diff_signed(double a, double b)
{
  a -= b;
  while( a <= -M_PI ) a += M_2__PI;
  while( a >   M_PI ) a -= M_2__PI;
  return a;
}


/*----------------------------------------------------------------------------*/
/*----------------------------- NFA computation ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x using the Lanczos approximation.
    See http://www.rskey.org/gamma.htm

    The formula used is
    @f[
      \Gamma(x) = \frac{ \sum_{n=0}^{N} q_n x^n }{ \Pi_{n=0}^{N} (x+n) }
                  (x+5.5)^{x+0.5} e^{-(x+5.5)}
    @f]
    so
    @f[
      \log\Gamma(x) = \log\left( \sum_{n=0}^{N} q_n x^n \right)
                      + (x+0.5) \log(x+5.5) - (x+5.5) - \sum_{n=0}^{N} \log(x+n)
    @f]
    and
      q0 = 75122.6331530,
      q1 = 80916.6278952,
      q2 = 36308.2951477,
      q3 = 8687.24529705,
      q4 = 1168.92649479,
      q5 = 83.8676043424,
      q6 = 2.50662827511.
 */
static double log_gamma_lanczos(double x)
{
  static double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
                         8687.24529705, 1168.92649479, 83.8676043424,
                         2.50662827511 };
  double a = (x+0.5) * log(x+5.5) - (x+5.5);
  double b = 0.0;
  int n;

  for(n=0;n<7;n++)
    {
      a -= log( x + (double) n );
      b += q[n] * pow( x, (double) n );
    }
  return a + log(b);
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x using Windschitl method.
    See http://www.rskey.org/gamma.htm

    The formula used is
    @f[
        \Gamma(x) = \sqrt{\frac{2\pi}{x}} \left( \frac{x}{e}
                    \sqrt{ x\sinh(1/x) + \frac{1}{810x^6} } \right)^x
    @f]
    so
    @f[
        \log\Gamma(x) = 0.5\log(2\pi) + (x-0.5)\log(x) - x
                      + 0.5x\log\left( x\sinh(1/x) + \frac{1}{810x^6} \right).
    @f]
    This formula is a good approximation when x > 15.
 */
static double log_gamma_windschitl(double x)
{
  return 0.918938533204673 + (x-0.5)*log(x) - x
         + 0.5*x*log( x*sinh(1/x) + 1/(810.0*pow(x,6.0)) );
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x. When x>15 use log_gamma_windschitl(),
    otherwise use log_gamma_lanczos().
 */
#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

/*----------------------------------------------------------------------------*/
/** Size of the table to store already computed inverse values.
 */
#define TABSIZE 100000

/*----------------------------------------------------------------------------*/
/** Computes -log10(NFA).

    NFA stands for Number of False Alarms:
    @f[
        \mathrm{NFA} = NT \cdot B(n,k,p)
    @f]

    - NT       - number of tests
    - B(n,k,p) - tail of binomial distribution with parameters n,k and p:
    @f[
        B(n,k,p) = \sum_{j=k}^n
                   \left(\begin{array}{c}n\\j\end{array}\right)
                   p^{j} (1-p)^{n-j}% alpha=2;
% alpha=2;
% 0.4906
% 0.0719
% alpha=2;
% 0.5513
% 0.0644
% longueur=4;
% eps=1/1;
% density=0.04;
% sizenum=256*sqrt(2);
% angth=22.5;
% p11=0.3934;
% p10=1-p11;
% p01=0.0863;
% p00=1-p01;
% p0=0.125;
% inputv=[alpha,longueur,eps,density,sizenum,angth,p11,p10,p01,p00,p0];
% I=double(I);
% lines1=mexlsd(I,inputv);
% size(lines1)
% lines1=lines1+1;
% [ML,NL]=size(lines1);
% lines2=reshape(lines1,[1,ML*NL]);
% lines3=reshape(lines2,[NL,ML]);
% lines=lines3';
% flagl=[];
% angleline=zeros(size(lines,1),1);
% minlen=100;
% maxlen=1;
% lines_20=lines;
% for il=1:size(lines,1)
%     lines_20(il,3)=lines(il,1);
%     ll=sqrt((lines(il,1)-lines(il,3))*(lines(il,1)-lines(il,3))+(lines(il,2)-lines(il,4))*(lines(il,2)-lines(il,4)));
%     angleline(il)=atan2(lines(il,2)-lines(il,4),lines(il,1)-lines(il,3));
%    if angleline(il)<0
%        angleline(il)=angleline(il)+pi;
%    end
%     if minlen>ll
%         minlen=ll;
%     end
%     if maxlen<ll
%         maxlen=ll;
%     end
% end
% [newangle,newindex]=sort(angleline);
% newlines=lines(newindex,:);
% minlen
% maxlen
% lines=newlines;
% figure,sarimshow(I),hold on
% max_len=0;
% for k=1:size(lines,1)
% xy=[lines(k,2),lines(k,1);lines(k,4),lines(k,3)];
% plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
% plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
% plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
% len=norm(lines(k,1:2)-lines(k,3:4));
% if(len>max_len)
%     max_len=len;
%     xy_long=xy;
% end
% end
% title('Markov-2-False-1178-min-7.14-max-36.9')
% length(lines)
    @f]

    The value -log10(NFA) is equivalent but more intuitive than NFA:
    - -1 corresponds to 10 mean false alarms
    -  0 corresponds to 1 mean false alarm
    -  1 corresponds to 0.1 mean false alarms
    -  2 corresponds to 0.01 mean false alarms
    -  ...

    Used this way, the bigger the value, better the detection,
    and a logarithmic scale is used.

    @param n,k,p binomial parameters.
    @param logNT logarithm of Number of Tests

    The computation is based in the gamma function by the following
    relation:
    @f[
        \left(\begin{array}{c}n\\k\end{array}\right)
        = \frac{ \Gamma(n+1) }{ \Gamma(k+1) \cdot \Gamma(n-k+1) }.
    @f]
    We use efficient algorithms to compute the logarithm of
    the gamma function.

    To make the computation faster, not all the sum is computed, part
    of the terms are neglected based on a bound to the error obtained
    (an error of 10% in the result is accepted).
 */
static double nfa(int n, int k, double p, double logNT,double *mnfa,int N)
{
   if(n>N||k>N)
        return 101;
   
  

  /* check parameters */
  if( n<0 || k<0 || k>n || p<=0.0 || p>=1.0 )
    error("nfa: wrong n, k or p values.");

  /* trivial cases */
  if( n<3 || k==0 ) return -logNT;
 
 

  
 
return -log10(mnfa[k*N+n])-logNT;

  
}


/*----------------------------------------------------------------------------*/
/*--------------------------- Rectangle structure ----------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Rectangle structure: line segment with width.
 */
struct rect
{
  double x1,y1,x2,y2;  /* first and second point of the line segment */
  double width;        /* rectangle width */
  double x,y;          /* center of the rectangle */
  double theta;        /* angle */
  double dx,dy;        /* (dx,dy) is vector oriented as the line segment */
  double prec;         /* tolerance angle */
  double p;            /* probability of a point with angle within 'prec' */
};

/*----------------------------------------------------------------------------*/
/** Copy one rectangle structure to another.
 */
static void rect_copy(struct rect * in, struct rect * out)
{
  /* check parameters */
  if( in == NULL || out == NULL ) error("rect_copy: invalid 'in' or 'out'.");

  /* copy values */
  out->x1 = in->x1;
  out->y1 = in->y1;
  out->x2 = in->x2;
  out->y2 = in->y2;
  out->width = in->width;
  out->x = in->x;
  out->y = in->y;
  out->theta = in->theta;
  out->dx = in->dx;
  out->dy = in->dy;
  out->prec = in->prec;
  out->p = in->p;
}

/*----------------------------------------------------------------------------*/
/** Rectangle points iterator.

    The integer coordinates of pixels inside a rectangle are
    iteratively explored. This structure keep track of the process and
    functions ri_ini(), ri_inc(), ri_end(), and ri_del() are used in
    the process. An example of how to use the iterator is as follows:
    \code

      struct rect * rec = XXX; // some rectangle
      rect_iter * i;
      for( i=ri_ini(rec); !ri_end(i); ri_inc(i) )
        {
          // your code, using 'i->x' and 'i->y' as coordinates
        }
      ri_del(i); // delete iterator

    \endcode
    The pixels are explored 'column' by 'column', where we call
    'column' a set of pixels with the same x value that are inside the
    rectangle. The following is an schematic representation of a
    rectangle, the 'column' being explored is marked by colons, and
    the current pixel being explored is 'x,y'.
    \verbatim

              vx[1],vy[1]
                 *   *
                *       *
               *           *
              *               ye
             *                :  *
        vx[0],vy[0]           :     *
               *              :        *
                  *          x,y          *
                     *        :              *
                        *     :            vx[2],vy[2]
                           *  :                *
        y                     ys              *
        ^                        *           *
        |                           *       *
        |                              *   *
        +---> x                      vx[3],vy[3]

    \endverbatim
    The first 'column' to be explored is the one with the smaller x
    value. Each 'column' is explored starting from the pixel of the
    'column' (inside the rectangle) with the smallest y value.

    The four corners of the rectangle are stored in order that rotates
    around the corners at the arrays 'vx[]' and 'vy[]'. The first
    point is always the one with smaller x value.

    'x' and 'y' are the coordinates of the pixel being explored. 'ys'
    and 'ye' are the start and end values of the current column being
    explored. So, 'ys' < 'ye'.
 */
typedef struct
{
  double vx[4];  /* rectangle's corner X coordinates in circular order */
  double vy[4];  /* rectangle's corner Y coordinates in circular order */
  double ys,ye;  /* start and end Y values of current 'column' */
  int x,y;       /* coordinates of currently explored pixel */
} rect_iter;

/*----------------------------------------------------------------------------*/
/** Interpolate y value corresponding to 'x' value given, in
    the line 'x1,y1' to 'x2,y2'; if 'x1=x2' return the smaller
    of 'y1' and 'y2'.

    The following restrictions are required:
    - x1 <= x2
    - x1 <= x
    - x  <= x2
 */
static double inter_low(double x, double x1, double y1, double x2, double y2)
{
  /* check parameters */
  if( x1 > x2 || x < x1 || x > x2 )
    error("inter_low: unsuitable input, 'x1>x2' or 'x<x1' or 'x>x2'.");

  /* interpolation */
  if( double_equal(x1,x2) && y1<y2 ) return y1;
  if( double_equal(x1,x2) && y1>y2 ) return y2;
  return y1 + (x-x1) * (y2-y1) / (x2-x1);
}

/*----------------------------------------------------------------------------*/
/** Interpolate y value corresponding to 'x' value given, in
    the line 'x1,y1' to 'x2,y2'; if 'x1=x2' return the larger
    of 'y1' and 'y2'.

    The following restrictions are required:
    - x1 <= x2
    - x1 <= x
    - x  <= x2
 */
static double inter_hi(double x, double x1, double y1, double x2, double y2)
{
  /* check parameters */
  if( x1 > x2 || x < x1 || x > x2 )
    error("inter_hi: unsuitable input, 'x1>x2' or 'x<x1' or 'x>x2'.");

  /* interpolation */
  if( double_equal(x1,x2) && y1<y2 ) return y2;
  if( double_equal(x1,x2) && y1>y2 ) return y1;
  return y1 + (x-x1) * (y2-y1) / (x2-x1);
}

/*----------------------------------------------------------------------------*/
/** Free memory used by a rectangle iterator.
 */
static void ri_del(rect_iter * iter)
{
  if( iter == NULL ) error("ri_del: NULL iterator.");
  free( (void *) iter );
}

/*----------------------------------------------------------------------------*/
/** Check if the iterator finished the full iteration.

    See details in \ref rect_iter
 */
static int ri_end(rect_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri_end: NULL iterator.");

  /* if the current x value is larger than the largest
     x value in the rectangle (vx[2]), we know the full
     exploration of the rectangle is finished. */
  return (double)(i->x) > i->vx[2];
}

/*----------------------------------------------------------------------------*/
/** Increment a rectangle iterator.

    See details in \ref rect_iter
 */
static void ri_inc(rect_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri_inc: NULL iterator.");

  /* if not at end of exploration,
     increase y value for next pixel in the 'column' */
  if( !ri_end(i) ) i->y++;

  /* if the end of the current 'column' is reached,
     and it is not the end of exploration,
     advance to the next 'column' */
  while( (double) (i->y) > i->ye && !ri_end(i) )
    {
      /* increase x, next 'column' */
      i->x++;

      /* if end of exploration, return */
      if( ri_end(i) ) return;

      /* update lower y limit (start) for the new 'column'.

         We need to interpolate the y value that corresponds to the
         lower side of the rectangle. The first thing is to decide if
         the corresponding side is

           vx[0],vy[0] to vx[3],vy[3] or
           vx[3],vy[3] to vx[2],vy[2]

         Then, the side is interpolated for the x value of the
         'column'. But, if the side is vertical (as it could happen if
         the rectangle is vertical and we are dealing with the first
         or last 'columns') then we pick the lower value of the side
         by using 'inter_low'.
       */
      if( (double) i->x < i->vx[3] )
        i->ys = inter_low((double)i->x,i->vx[0],i->vy[0],i->vx[3],i->vy[3]);
      else
        i->ys = inter_low((double)i->x,i->vx[3],i->vy[3],i->vx[2],i->vy[2]);

      /* update upper y limit (end) for the new 'column'.

         We need to interpolate the y value that corresponds to the
         upper side of the rectangle. The first thing is to decide if
         the corresponding side is

           vx[0],vy[0] to vx[1],vy[1] or
           vx[1],vy[1] to vx[2],vy[2]

         Then, the side is interpolated for the x value of the
         'column'. But, if the side is vertical (as it could happen if
         the rectangle is vertical and we are dealing with the first
         or last 'columns') then we pick the lower value of the side
         by using 'inter_low'.
       */
      if( (double)i->x < i->vx[1] )
        i->ye = inter_hi((double)i->x,i->vx[0],i->vy[0],i->vx[1],i->vy[1]);
      else
        i->ye = inter_hi((double)i->x,i->vx[1],i->vy[1],i->vx[2],i->vy[2]);

      /* new y */
      i->y = (int) ceil(i->ys);
    }
}

/*----------------------------------------------------------------------------*/
/** Create and initialize a rectangle iterator.

    See details in \ref rect_iter
 */
static rect_iter * ri_ini(struct rect * r)
{
  double vx[4],vy[4];
  int n,offset;
  rect_iter * i;

  /* check parameters */
  if( r == NULL ) error("ri_ini: invalid rectangle.");

  /* get memory */
  i = (rect_iter *) malloc(sizeof(rect_iter));
  if( i == NULL ) error("ri_ini: Not enough memory.");

  /* build list of rectangle corners ordered
     in a circular way around the rectangle */
  vx[0] = r->x1 - r->dy * r->width / 2.0;
  vy[0] = r->y1 + r->dx * r->width / 2.0;
  vx[1] = r->x2 - r->dy * r->width / 2.0;
  vy[1] = r->y2 + r->dx * r->width / 2.0;
  vx[2] = r->x2 + r->dy * r->width / 2.0;
  vy[2] = r->y2 - r->dx * r->width / 2.0;
  vx[3] = r->x1 + r->dy * r->width / 2.0;
  vy[3] = r->y1 - r->dx * r->width / 2.0;

  /* compute rotation of index of corners needed so that the first
     point has the smaller x.

     if one side is vertical, thus two corners have the same smaller x
     value, the one with the largest y value is selected as the first.
   */
  if( r->x1 < r->x2 && r->y1 <= r->y2 ) offset = 0;
  else if( r->x1 >= r->x2 && r->y1 < r->y2 ) offset = 1;
  else if( r->x1 > r->x2 && r->y1 >= r->y2 ) offset = 2;
  else offset = 3;

  /* apply rotation of index. */
  for(n=0; n<4; n++)
    {
      i->vx[n] = vx[(offset+n)%4];
      i->vy[n] = vy[(offset+n)%4];
    }

  /* Set an initial condition.

     The values are set to values that will cause 'ri_inc' (that will
     be called immediately) to initialize correctly the first 'column'
     and compute the limits 'ys' and 'ye'.

     'y' is set to the integer value of vy[0], the starting corner.

     'ys' and 'ye' are set to very small values, so 'ri_inc' will
     notice that it needs to start a new 'column'.

     The smallest integer coordinate inside of the rectangle is
     'ceil(vx[0])'. The current 'x' value is set to that value minus
     one, so 'ri_inc' (that will increase x by one) will advance to
     the first 'column'.
   */
  i->x = (int) ceil(i->vx[0]) - 1;
  i->y = (int) ceil(i->vy[0]);
  i->ys = i->ye = -DBL_MAX;

  /* advance to the first pixel */
  ri_inc(i);

  return i;
}

/*----------------------------------------------------------------------------*/
/** Compute a rectangle's NFA value.
 */
static double rect_nfa(struct rect * rec, image_double angles, double logNT,double *image,int N,int minreg)
{
  rect_iter * i;
  int pts = 0;
  int alg = 0;

  /* check parameters */
  if( rec == NULL ) error("rect_nfa: invalid rectangle.");
  if( angles == NULL ) error("rect_nfa: invalid 'angles'.");

  /* compute the total number of pixels and of aligned points in 'rec' */
  for(i=ri_ini(rec); !ri_end(i); ri_inc(i)) /* rectangle iterator */
    if( i->x >= 0 && i->y >= 0 &&
        i->x < (int) angles->xsize && i->y < (int) angles->ysize )
      {
        ++pts; /* total number of pixels counter */
        if( isaligned(i->x, i->y, angles, rec->theta, rec->prec) )
          ++alg; /* aligned points counter */
      }
  ri_del(i); /* delete iterator */
  if(pts<minreg)
      return -1;
  else
  return nfa(pts,alg,rec->p,logNT,image,N); /* compute NFA value */
}


/*----------------------------------------------------------------------------*/
/*---------------------------------- Regions ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Compute region's angle as the principal inertia axis of the region.

    The following is the region inertia matrix A:
    @f[

        A = \left(\begin{array}{cc}
                                    Ixx & Ixy \\
                                    Ixy & Iyy \\
             \end{array}\right)

    @f]
    where

      Ixx =   sum_i G(i).(y_i - cx)^2

      Iyy =   sum_i G(i).(x_i - cy)^2

      Ixy = - sum_i G(i).(x_i - cx).(y_i - cy)

    and
    - G(i) is the gradient norm at pixel i, used as pixel's weight.
    - x_i and y_i are the coordinates of pixel i.
    - cx and cy are the coordinates of the center of th region.

    lambda1 and lambda2 are the eigenvalues of matrix A,
    with lambda1 >= lambda2. They are found by solving the
    characteristic polynomial:

      det( lambda I - A) = 0

    that gives:

      lambda1 = ( Ixx + Iyy + sqrt( (Ixx-Iyy)^2 + 4.0*Ixy*Ixy) ) / 2

      lambda2 = ( Ixx + Iyy - sqrt( (Ixx-Iyy)^2 + 4.0*Ixy*Ixy) ) / 2

    To get the line segment direction we want to get the angle the
    eigenvector associated to the smallest eigenvalue. We have
    to solve for a,b in:

      a.Ixx + b.Ixy = a.lambda2

      a.Ixy + b.Iyy = b.lambda2

    We want the angle theta = atan(b/a). It can be computed with
    any of the two equations:

      theta = atan( (lambda2-Ixx) / Ixy )

    or

      theta = atan( Ixy / (lambda2-Iyy) )

    When |Ixx| > |Iyy| we use the first, otherwise the second (just to
    get better numeric precision).
 */
static double get_theta( struct point * reg, int reg_size, double x, double y,
                         image_double modgrad, double reg_angle, double prec )
{
  double lambda,theta,weight;
  double Ixx = 0.0;
  double Iyy = 0.0;
  double Ixy = 0.0;
  int i;

  /* check parameters */
  if( reg == NULL ) error("get_theta: invalid region.");
  if( reg_size <= 1 ) error("get_theta: region size <= 1.");
  if( modgrad == NULL || modgrad->data == NULL )
    error("get_theta: invalid 'modgrad'.");
  if( prec < 0.0 ) error("get_theta: 'prec' must be positive.");

  /* compute inertia matrix */
  for(i=0; i<reg_size; i++)
    {
      weight = modgrad->data[ reg[i].x + reg[i].y * modgrad->xsize ];
      Ixx += ( (double) reg[i].y - y ) * ( (double) reg[i].y - y ) * weight;
      Iyy += ( (double) reg[i].x - x ) * ( (double) reg[i].x - x ) * weight;
      Ixy -= ( (double) reg[i].x - x ) * ( (double) reg[i].y - y ) * weight;
    }
  if( double_equal(Ixx,0.0) && double_equal(Iyy,0.0) && double_equal(Ixy,0.0) )
    error("get_theta: null inertia matrix.");

  /* compute smallest eigenvalue */
  lambda = 0.5 * ( Ixx + Iyy - sqrt( (Ixx-Iyy)*(Ixx-Iyy) + 4.0*Ixy*Ixy ) );

  /* compute angle */
  theta = fabs(Ixx)>fabs(Iyy) ? atan2(lambda-Ixx,Ixy) : atan2(Ixy,lambda-Iyy);

  /* The previous procedure doesn't cares about orientation,
     so it could be wrong by 180 degrees. Here is corrected if necessary. */
  if( angle_diff(theta,reg_angle) > prec ) theta += M_PI;

  return theta;
}

/*----------------------------------------------------------------------------*/
/** Computes a rectangle that covers a region of points.
 */
static void region2rect( struct point * reg, int reg_size,
                         image_double modgrad, double reg_angle,
                         double prec, double p, struct rect * rec )
{
  double x,y,dx,dy,l,w,theta,weight,sum,l_min,l_max,w_min,w_max;
  int i;

  /* check parameters */
  if( reg == NULL ) error("region2rect: invalid region.");
  if( reg_size <= 1 ) error("region2rect: region size <= 1.");
  if( modgrad == NULL || modgrad->data == NULL )
    error("region2rect: invalid image 'modgrad'.");
  if( rec == NULL ) error("region2rect: invalid 'rec'.");

  /* center of the region:

     It is computed as the weighted sum of the coordinates
     of all the pixels in the region. The norm of the gradient
     is used as the weight of a pixel. The sum is as follows:
       cx = \sum_i G(i).x_i
       cy = \sum_i G(i).y_i
     where G(i) is the norm of the gradient of pixel i
     and x_i,y_i are its coordinates.
   */
  x = y = sum = 0.0;
  for(i=0; i<reg_size; i++)
    {
      weight = modgrad->data[ reg[i].x + reg[i].y * modgrad->xsize ];
      x += (double) reg[i].x * weight;
      y += (double) reg[i].y * weight;
      sum += weight;
    }
  if( sum <= 0.0 ) error("region2rect: weights sum equal to zero.");
  x /= sum;
  y /= sum;

  /* theta */
  theta = get_theta(reg,reg_size,x,y,modgrad,reg_angle,prec);

  /* length and width:

     'l' and 'w' are computed as the distance from the center of the
     region to pixel i, projected along the rectangle axis (dx,dy) and
     to the orthogonal axis (-dy,dx), respectively.

     The length of the rectangle goes from l_min to l_max, where l_min
     and l_max are the minimum and maximum values of l in the region.
     Analogously, the width is selected from w_min to w_max, where
     w_min and w_max are the minimum and maximum of w for the pixels
     in the region.
   */
  dx = cos(theta);
  dy = sin(theta);
  l_min = l_max = w_min = w_max = 0.0;
  for(i=0; i<reg_size; i++)
    {
      l =  ( (double) reg[i].x - x) * dx + ( (double) reg[i].y - y) * dy;
      w = -( (double) reg[i].x - x) * dy + ( (double) reg[i].y - y) * dx;

      if( l > l_max ) l_max = l;
      if( l < l_min ) l_min = l;
      if( w > w_max ) w_max = w;
      if( w < w_min ) w_min = w;
    }

  /* store values */
  rec->x1 = x + l_min * dx;
  rec->y1 = y + l_min * dy;
  rec->x2 = x + l_max * dx;
  rec->y2 = y + l_max * dy;
  rec->width = w_max - w_min;
  rec->x = x;
  rec->y = y;
  rec->theta = theta;
  rec->dx = dx;
  rec->dy = dy;
  rec->prec = prec;
  rec->p = p;

  /* we impose a minimal width of one pixel

     A sharp horizontal or vertical step would produce a perfectly
     horizontal or vertical region. The width computed would be
     zero. But that corresponds to a one pixels width transition in
     the image.
   */
  if( rec->width < 1.0 ) rec->width = 1.0;
}

/*----------------------------------------------------------------------------*/
/** Build a region of pixels that share the same angle, up to a
    tolerance 'prec', starting at point (x,y).
 */
static void region_grow( int x, int y, image_double angles, struct point * reg,
                         int * reg_size, double * reg_angle, image_char used,
                         double prec )
{
  double sumdx,sumdy;
  int xx,yy,i;

  /* check parameters */
  if( x < 0 || y < 0 || x >= (int) angles->xsize || y >= (int) angles->ysize )
    error("region_grow: (x,y) out of the image.");
  if( angles == NULL || angles->data == NULL )
    error("region_grow: invalid image 'angles'.");
  if( reg == NULL ) error("region_grow: invalid 'reg'.");
  if( reg_size == NULL ) error("region_grow: invalid pointer 'reg_size'.");
  if( reg_angle == NULL ) error("region_grow: invalid pointer 'reg_angle'.");
  if( used == NULL || used->data == NULL )
    error("region_grow: invalid image 'used'.");

  /* first point of the region */
  *reg_size = 1;
  reg[0].x = x;
  reg[0].y = y;
  *reg_angle = angles->data[x+y*angles->xsize];  /* region's angle */
  sumdx = cos(*reg_angle);
  sumdy = sin(*reg_angle);
  used->data[x+y*used->xsize] = USED;

  /* try neighbors as new region points */
  for(i=0; i<*reg_size; i++)
    for(xx=reg[i].x-1; xx<=reg[i].x+1; xx++)
      for(yy=reg[i].y-1; yy<=reg[i].y+1; yy++)
        if( xx>=0 && yy>=0 && xx<(int)used->xsize && yy<(int)used->ysize &&
            used->data[xx+yy*used->xsize] != USED &&
            isaligned(xx,yy,angles,*reg_angle,prec) )
          {
            /* add point */
            used->data[xx+yy*used->xsize] = USED;
            reg[*reg_size].x = xx;
            reg[*reg_size].y = yy;
            ++(*reg_size);

            /* update region's angle */
            sumdx += cos( angles->data[xx+yy*angles->xsize] );
            sumdy += sin( angles->data[xx+yy*angles->xsize] );
            *reg_angle = atan2(sumdy,sumdx);
          }
}

/*----------------------------------------------------------------------------*/
/** Try some rectangles variations to improve NFA value. Only if the
    rectangle is not meaningful (i.e., log_nfa <= log_eps).
 */
static double rect_improve( struct rect * rec, image_double angles,
                            double logNT, double log_eps,double* mnfa,double* mnfa_2,double* mnfa_4,int Nnfa,int minsize, int minsize2,int minsize4 )
{
  struct rect r;
  double log_nfa,log_nfa_new;
  double delta = 0.5;
  double delta_2 = delta / 2.0;
  int n;
  rect_copy(rec,&r);
 if(r.p>0.1)
      log_nfa = rect_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
 else if(r.p>0.05)
           log_nfa= rect_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa = rect_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
  

  if( log_nfa > log_eps ) return log_nfa;

  /* try finer precisions */
  rect_copy(rec,&r);
  for(n=0; n<1; n++)
    {
      r.p /= 2.0;
      r.prec = r.p * M_PI;
      if(r.p>0.1)
      log_nfa_new = rect_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
      if( log_nfa_new > log_nfa )
        {
          log_nfa = log_nfa_new;
          rect_copy(&r,rec);
        }
    }

  if( log_nfa > log_eps ) return log_nfa;
  /* try to reduce width */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width - delta) >= 0.5 )
        {
          r.width -= delta;
           if(r.p>0.1)
      log_nfa_new = rect_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
          
          if( log_nfa_new > log_nfa )
            {
              rect_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try to reduce one side of the rectangle */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width - delta) >= 0.5 )
        {
          r.x1 += -r.dy * delta_2;
          r.y1 +=  r.dx * delta_2;
          r.x2 += -r.dy * delta_2;
          r.y2 +=  r.dx * delta_2;
          r.width -= delta;
           
         if(r.p>0.1)
      log_nfa_new = rect_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
         
          if( log_nfa_new > log_nfa )
            {
              rect_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try to reduce the other side of the rectangle */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width - delta) >= 0.5 )
        {
          r.x1 -= -r.dy * delta_2;
          r.y1 -=  r.dx * delta_2;
          r.x2 -= -r.dy * delta_2;
          r.y2 -=  r.dx * delta_2;
          r.width -= delta;
         if(r.p>0.1)
      log_nfa_new = rect_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
          if( log_nfa_new > log_nfa )
            {
              rect_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try even finer precisions */
  rect_copy(rec,&r);
  for(n=0; n<1; n++)
    {
      r.p /= 2.0;
      r.prec = r.p * M_PI;
      if(r.p>0.1)
      log_nfa_new = rect_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
      if( log_nfa_new > log_nfa )
        {
          log_nfa = log_nfa_new;
          rect_copy(&r,rec);
        }
    }

  return log_nfa;
}

/*----------------------------------------------------------------------------*/
/** Reduce the region size, by elimination the points far from the
    starting point, until that leads to rectangle with the right
    density of region points or to discard the region if too small.
 */
static int reduce_region_radius( struct point * reg, int * reg_size,
                                 image_double modgrad, double reg_angle,
                                 double prec, double p, struct rect * rec,
                                 image_char used, image_double angles,
                                 double density_th )
{
  double density,rad1,rad2,rad,xc,yc;
  int i;

  /* check parameters */
  if( reg == NULL ) error("reduce_region_radius: invalid pointer 'reg'.");
  if( reg_size == NULL )
    error("reduce_region_radius: invalid pointer 'reg_size'.");
  if( prec < 0.0 ) error("reduce_region_radius: 'prec' must be positive.");
  if( rec == NULL ) error("reduce_region_radius: invalid pointer 'rec'.");
  if( used == NULL || used->data == NULL )
    error("reduce_region_radius: invalid image 'used'.");
  if( angles == NULL || angles->data == NULL )
    error("reduce_region_radius: invalid image 'angles'.");

  /* compute region points density */
  density = (double) *reg_size /
                         ( dist(rec->x1,rec->y1,rec->x2,rec->y2) * rec->width );

  /* if the density criterion is satisfied there is nothing to do */
  if( density >= density_th ) return TRUE;

  /* compute region's radius */
  xc = (double) reg[0].x;
  yc = (double) reg[0].y;
  rad1 = dist( xc, yc, rec->x1, rec->y1 );
  rad2 = dist( xc, yc, rec->x2, rec->y2 );
  rad = rad1 > rad2 ? rad1 : rad2;

  /* while the density criterion is not satisfied, remove farther pixels */
  while( density < density_th )
    {
      rad *= 0.75; /* reduce region's radius to 75% of its value */

      /* remove points from the region and update 'used' map */
      for(i=0; i<*reg_size; i++)
        if( dist( xc, yc, (double) reg[i].x, (double) reg[i].y ) > rad )
          {
            /* point not kept, mark it as NOTUSED */
            used->data[ reg[i].x + reg[i].y * used->xsize ] = NOTUSED;
            /* remove point from the region */
            reg[i].x = reg[*reg_size-1].x; /* if i==*reg_size-1 copy itself */
            reg[i].y = reg[*reg_size-1].y;
            --(*reg_size);
            --i; /* to avoid skipping one point */
          }

      /* reject if the region is too small.
         2 is the minimal region size for 'region2rect' to work. */
      if( *reg_size < 2 ) return FALSE;

      /* re-compute rectangle */
      region2rect(reg,*reg_size,modgrad,reg_angle,prec,p,rec);

      /* re-compute region points density */
      density = (double) *reg_size /
                         ( dist(rec->x1,rec->y1,rec->x2,rec->y2) * rec->width );
    }

  /* if this point is reached, the density criterion is satisfied */
  return TRUE;
}

/*----------------------------------------------------------------------------*/
/** Refine a rectangle.

    For that, an estimation of the angle tolerance is performed by the
    standard deviation of the angle at points near the region's
    starting point. Then, a new region is grown starting from the same
    point, but using the estimated angle tolerance. If this fails to
    produce a rectangle with the right density of region points,
    'reduce_region_radius' is called to try to satisfy this condition.
 */
static int refine( struct point * reg, int * reg_size, image_double modgrad,
                   double reg_angle, double prec, double p, struct rect * rec,
                   image_char used, image_double angles, double density_th )
{
  double angle,ang_d,mean_angle,tau,density,xc,yc,ang_c,sum,s_sum;
  int i,n;

  /* check parameters */
  if( reg == NULL ) error("refine: invalid pointer 'reg'.");
  if( reg_size == NULL ) error("refine: invalid pointer 'reg_size'.");
  if( prec < 0.0 ) error("refine: 'prec' must be positive.");
  if( rec == NULL ) error("refine: invalid pointer 'rec'.");
  if( used == NULL || used->data == NULL )
    error("refine: invalid image 'used'.");
  if( angles == NULL || angles->data == NULL )
    error("refine: invalid image 'angles'.");

  /* compute region points density */
  density = (double) *reg_size /
                         ( dist(rec->x1,rec->y1,rec->x2,rec->y2) * rec->width );

  /* if the density criterion is satisfied there is nothing to do */
  if( density >= density_th ) return TRUE;

  /*------ First try: reduce angle tolerance ------*/

  /* compute the new mean angle and tolerance */
  xc = (double) reg[0].x;
  yc = (double) reg[0].y;
  ang_c = angles->data[ reg[0].x + reg[0].y * angles->xsize ];
  sum = s_sum = 0.0;
  n = 0;
  for(i=0; i<*reg_size; i++)
    {
      used->data[ reg[i].x + reg[i].y * used->xsize ] = NOTUSED;
      if( dist( xc, yc, (double) reg[i].x, (double) reg[i].y ) < rec->width )
        {
          angle = angles->data[ reg[i].x + reg[i].y * angles->xsize ];
          ang_d = angle_diff_signed(angle,ang_c);
          sum += ang_d;
          s_sum += ang_d * ang_d;
          ++n;
        }
    }
  mean_angle = sum / (double) n;
  tau = 2.0 * sqrt( (s_sum - 2.0 * mean_angle * sum) / (double) n
                         + mean_angle*mean_angle ); /* 2 * standard deviation */

  /* find a new region from the same starting point and new angle tolerance */
  tau=prec/2.0;
  region_grow(reg[0].x,reg[0].y,angles,reg,reg_size,&reg_angle,used,tau);
/*prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;*/
  p=tau/M_PI;
  /* if the region is too small, reject */
  if( *reg_size < 2 ) return FALSE;

  /* re-compute rectangle */
  region2rect(reg,*reg_size,modgrad,reg_angle,tau,p,rec);

  /* re-compute region points density */
  density = (double) *reg_size /
                      ( dist(rec->x1,rec->y1,rec->x2,rec->y2) * rec->width );

  /*------ Second try: reduce region radius ------*/
  if( density < density_th )
    return reduce_region_radius( reg, reg_size, modgrad, reg_angle, prec, p,
                                 rec, used, angles, density_th );

  /* if this point is reached, the density criterion is satisfied */
  return TRUE;
}
static void NFA_matrix(double *output,double p0,double p11,double p01,int N)
{
    double p1=1.0-p0;
    double p10=1.0-p11;
    double p00=1.0-p01;
     
  double *plk0;
  plk0=(double *) malloc(N*N*sizeof(double));
  double *plk1;
  plk1=(double *) malloc(N*N*sizeof(double));
  double *output0;
  output0=(double *) malloc(N*sizeof(double));
  double *output1;
 
  output1=(double *) malloc(N*sizeof(double));
  int i,j;
  for(i=0;i<N;i++)
    {
      for(j=0;j<N;j++)
	{
	  plk0[i*N+j]=0;
	  plk1[i*N+j]=0;
	  output[i*N+j]=0;
	}
      
      output0[i]=0;
      output1[i]=0;
    }

  for(i=0;i<N;i++)
  for (j=0;j<N;j++)
    {
      
      if(i==0)
	{
	  plk0[0+j]=1;
	  plk1[0+j]=1;
	}
      else if(i==1)
	{
	  plk0[i*N+j]=p01;
	  plk1[i*N+j]=p11;
	}
      else
	{
	  plk0[i*N+j]=0;
	  plk1[i*N+j]=0;
	}
    }
  
  
  for(i=1;i<j;i++)
    {
      for(j=2;j<N;j++)
	{
	  plk0[i*N+j]=plk0[i*N+j-1]*p00+plk1[(i-1)*N+j-1]*p01;
	  plk1[i*N+j]=plk0[i*N+j-1]*p10+plk1[(i-1)*N+j-1]*p11;
	}
    }

  
  
  
 
  for(i=1;i<j;i++)
    {
      for(j=3;j<N;j++)
	{
             

	  
          output[i*N+j]=(plk0[i*N+j-1]*p0+plk1[(i-1)*N+j-1]*p1);
         
          
         
	 
	  
	}
      
     
    }
  
  free((void *) plk0);
  free((void *) plk1);
  free((void *) output0);
  free((void *) output1);
}

/*----------------------------------------------------------------------------*/
/*--------------------- Markov Transition Probabilities ----------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Given a conditioning image, estimate the Markov transition probabilities  for 
 *  Gradient-by-Ratio computation, P(1|1) and P(1|0).  
 */
static void make_markov( double * img, int X, int Y,
                           double ang_th, int n_bins,
                           double * inputv)
{

  fprintf(stdout,"MakeMarkov started\n");
  fflush(stdout);

  //Instantiate variables per LSDSAR
  image_double image;
  image_double angles,modgrad;
  struct coorlist * list_p;
  void * mem_p;
  unsigned int xsize,ysize;
  double prec,p;
  /* check parameters */
  if( img == NULL || X <= 0 || Y <= 0 ) error("invalid image input.");
  if( ang_th <= 0.0 || ang_th >= 180.0 ) error("'ang_th' value must be in the range (0,180).");
  if( n_bins <= 0 ) error("'n_bins' value must be positive.");
  /* angle tolerance */
  prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;
  double beta;
  beta=inputv[0];
  /* load and scale image (if necessary) and compute angle at each pixel */
  image = new_image_double_ptr( (unsigned int) X, (unsigned int) Y, img );
  angles = ll_angle( image, &list_p, &mem_p, &modgrad, (unsigned int) n_bins,beta);
  xsize = angles->xsize;
  ysize = angles->ysize;
  
 //INDIVIDUAL LINE FREQUENCIES 
 int ang_mult, xx, yy;
 double hp_x0,hp_10,hp_x1,hp_11;
 double vp_x0,vp_10,vp_x1,vp_11;
 double hp_1,vp_1,hp_0,vp_0;
 int x_t, x_tminus;

 /* Calculate horizontal and vertical alignment freqencies*/
 // iterate precision options (prec, prec/2, prec/4)
 for(ang_mult=5;ang_mult<=9;ang_mult+=2)
 {
     hp_1=0; hp_0=0; vp_1=0;  vp_0=0;
	 
	 // iterate vertical lines
	 for(xx=0;xx<xsize;xx+=1)
	 {
	         vp_x0 = 0; vp_10 = 0; vp_x1 = 0; vp_11 = 0;
		 for(yy=0;yy<ysize-1;yy+=1)
		 {
				x_t        = isaligned(xx,  yy+1,  angles,0,prec);
				x_tminus   = isaligned(xx,  yy,    angles,0,prec);
				
				if ( x_tminus==0 ) {++vp_x0; if ( x_t==1 ) ++vp_10;}
				else {++vp_x1; if ( x_t==1 ) ++vp_11;}
		  }
		  if( vp_x1>0) vp_1+=vp_11/vp_x1;
	  	  vp_0+=vp_10/vp_x0;
	  }

	 // iterate horizontal lines 
	 for(yy=0;yy<ysize;yy+=1)
	 {
		 hp_x0 = 0; hp_10 = 0; hp_x1 = 0; hp_11 = 0;
		 for(xx=0;xx<xsize-1;xx+=1)
		 {
				x_t      = isaligned(xx+1,  yy,  angles,M_PI/2.,   prec);
				x_tminus = isaligned(xx,    yy,  angles,M_PI/2.,   prec);
				
				if ( x_tminus==0 ){++hp_x0; if ( x_t==1 ) ++hp_10;}
				else {++hp_x1; if ( x_t==1 ) ++hp_11;}				
	 	  }
	 	  if( hp_x1>0) hp_1+=hp_11/hp_x1;
	          hp_0+=hp_10/hp_x0;
	}

	 //Catch extrema cases 
	 inputv[ang_mult]   = (hp_1 + vp_1)/(xsize+ysize);
	 if(inputv[ang_mult]<=0) inputv[ang_mult]=0.0001;
	 if(inputv[ang_mult]>=1) inputv[ang_mult]=0.9999;
	 inputv[ang_mult+1] = (hp_0 + vp_0)/(xsize+ysize);
	 if(inputv[ang_mult+1]<=0) inputv[ang_mult+1]=0.0001;
	 if(inputv[ang_mult+1]>=1) inputv[ang_mult+1]=0.9999;
      
	 // reduce tolerance for next loop 
	 prec/=2;
	 
 }
  
  free( (void *) image );   /* only the double_image structure should be freed,
                               the data pointer was provided to this functions
                              and should not be destroyed.                 */
  free_image_double(angles);
}


/*----------------------------------------------------------------------------*/
/*-------------------------- Line Segment Detector ---------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** LSD full interface.
 */
double * LineSegmentDetection( int * n_out,
                               double * img, int X, int Y,
                               double ang_th, double log_eps, double density_th,
                               int n_bins,
                               int ** reg_img, int * reg_x, int * reg_y ,double * inputv)
{
  image_double image;
  ntuple_list out = new_ntuple_list(7);
  double * return_value;
  image_double angles,modgrad;
  image_char used;
  image_int region = NULL;
  struct coorlist * list_p;
  void * mem_p;
  struct rect rec;
  struct point * reg;
  int reg_size,min_reg_size,i;
  unsigned int xsize,ysize;
  double reg_angle,prec,p,log_nfa,logNT;
  int ls_count = 0;                   /* line segments are numbered 1,2,3,... */


  /* check parameters */
  if( img == NULL || X <= 0 || Y <= 0 ) error("invalid image input.");
 
  if( ang_th <= 0.0 || ang_th >= 180.0 )
    error("'ang_th' value must be in the range (0,180).");
  if( density_th < 0.0 || density_th > 1.0 )
    error("'density_th' value must be in the range [0,1].");
  if( n_bins <= 0 ) error("'n_bins' value must be positive.");


  /* angle tolerance */
  prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;
 
 
double beta;
beta=inputv[0];
int sizenum;
sizenum=(int) inputv[3];
  /* load and scale image (if necessary) and compute angle at each pixel */
  image = new_image_double_ptr( (unsigned int) X, (unsigned int) Y, img );

    angles = ll_angle( image, &list_p, &mem_p, &modgrad,
                       (unsigned int) n_bins,beta);
  xsize = angles->xsize;
  ysize = angles->ysize;

  /* Number of Tests - NT

     The theoretical number of tests is Np.(XY)^(5/2)
     where X and Y are number of columns and rows of the image.
     Np corresponds to the number of angle precisions considered.
     As the procedure 'rect_improve' tests 5 times to halve the
     angle precision, and 5 more times after improving other factors,
     11 different precision values are potentially tested. Thus,
     the number of tests is
       11 * (X*Y)^(5/2)
     whose logarithm value is
       log10(11) + 5/2 * (log10(X) + log10(Y)).
  */
 
  /* initialize some structures */
  if( reg_img != NULL && reg_x != NULL && reg_y != NULL ) /* save region data */
    region = new_image_int_ini(angles->xsize,angles->ysize,0);
  used = new_image_char_ini(xsize,ysize,NOTUSED);
  reg = (struct point *) calloc( (size_t) (xsize*ysize), sizeof(struct point) );
  if( reg == NULL ) error("not enough memory!");
double p0,p1;
  p1=p;
  p0=1-p1;
  
  double p11=inputv[5];
 
  double p10=1.0-p11;
 
  
  double p01;
  p01=inputv[6];
 
  double p00;
  p00=1.0-p01;
   logNT = 5.0 * ( log10( (double) xsize ) + log10( (double) ysize ) ) / 2.0
          + log10(3.0);
   
                                   
min_reg_size=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
 
 int N=sizenum;
  double *output;
  output=( double *) malloc(N*N*sizeof(double));
  int NOUT=N;
  NFA_matrix(output,p0,p11,p01,N);
  double *output_2;
  output_2=( double *) malloc(N*N*sizeof(double));
 p11=inputv[7];
 p01=inputv[8];
p1=p/2;
p0=1-p1;
int min_reg_size_2;
min_reg_size_2=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
NFA_matrix(output_2,p0,p11,p01,N);
  double *output_4;
  output_4=( double *) malloc(N*N*sizeof(double));
  p11=inputv[9];
  p01=inputv[10];
p1=p/4;
p0=1-p1;
int min_reg_size_4;
min_reg_size_4=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
NFA_matrix(output_4,p0,p11,p01,N);

  /* search for line segments */
  for(; list_p != NULL; list_p = list_p->next )
    if( used->data[ list_p->x + list_p->y * used->xsize ] == NOTUSED &&
        angles->data[ list_p->x + list_p->y * angles->xsize ] != NOTDEF )
       /* there is no risk of double comparison problems here
          because we are only interested in the exact NOTDEF value */
      {
        /* find the region of connected point and ~equal angle */
        region_grow( list_p->x, list_p->y, angles, reg, &reg_size,
                     &reg_angle, used, prec );

        /* reject small regions */
        if( reg_size < min_reg_size ) continue;

        /* construct rectangular approximation for the region */
        region2rect(reg,reg_size,modgrad,reg_angle,prec,p,&rec);

        /* Check if the rectangle exceeds the minimal density of
           region points. If not, try to improve the region.
           The rectangle will be rejected if the final one does
           not fulfill the minimal density condition.
           This is an addition to the original LSD algorithm published in
           "LSD: A Fast Line Segment Detector with a False Detection Control"
           by R. Grompone von Gioi, J. Jakubowicz, J.M. Morel, and G. Randall.
           The original algorithm is obtained with density_th = 0.0.
         */
        if( !refine( reg, &reg_size, modgrad, reg_angle,
                     prec, p, &rec, used, angles, density_th ) ) continue;

        /* compute NFA value */
        
  
 

        log_nfa = rect_improve(&rec,angles,logNT,log_eps,output,output_2,output_4,NOUT,min_reg_size,min_reg_size_2,min_reg_size_4);
         
        if( log_nfa <= log_eps ) continue;

        /* A New Line Segment was found! */
        ++ls_count;  /* increase line segment counter */

        /*
           The gradient was computed with a 2x2 mask, its value corresponds to
           points with an offset of (0.5,0.5), that should be added to output.
           The coordinates origin is at the center of pixel (0,0).
         */
        rec.x1 += 0.; rec.y1 += 0.;
        rec.x2 += 0.; rec.y2 += 0.;

      

        /* add line segment found to output */
        add_7tuple( out, rec.x1, rec.y1, rec.x2, rec.y2,
                         rec.width, rec.p, log_nfa );

        /* add region number to 'region' image if needed */
        if( region != NULL )
          for(i=0; i<reg_size; i++)
            region->data[ reg[i].x + reg[i].y * region->xsize ] = ls_count;
      }


  /* free memory */
  free( (void *) image );   /* only the double_image structure should be freed,
                               the data pointer was provided to this functions
                              and should not be destroyed.                 */
  free_image_double(angles);
  free_image_double(modgrad);
  free_image_char(used);
  free( (void *) reg );
  free( (void *) mem_p );
free((void *) output); 
free((void *) output_2);
free((void *) output_4);
  /* return the result */
  if( reg_img != NULL && reg_x != NULL && reg_y != NULL )
    {
      if( region == NULL ) error("'region' should be a valid image.");
      *reg_img = region->data;
      if( region->xsize > (unsigned int) INT_MAX ||
          region->xsize > (unsigned int) INT_MAX )
        error("region image to big to fit in INT sizes.");
      *reg_x = (int) (region->xsize);
      *reg_y = (int) (region->ysize);

      /* free the 'region' structure.
         we cannot use the function 'free_image_int' because we need to keep
         the memory with the image data to be returned by this function. */
      free( (void *) region );
    }
  if( out->size > (unsigned int) INT_MAX )
    error("too many detections to fit in an INT.");
  *n_out = (int) (out->size);

  return_value = out->values;
  free( (void *) out );  /* only the 'ntuple_list' structure must be freed,
                            but the 'values' pointer must be keep to return
                            as a result. */

  return return_value;
}



/*----------------------------------------------------------------------------*/
/*---------------- CONDITIONING LINE SEGMENT DETECTOR ------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**  Construct NFA table for I0 rectangles with parameters (width,length,orientation)
 *   Fill out interpolant surface.
 *   Normalize surface as a valid probability space 
 *   Application of conditioning P(I0) provided in secondary function.  
 */
/* Get all rectangle NFA data in I0. */
double * Conditional_LineSegmentDetection( int * n_out,
                                 double * img, int X, int Y,
                                 double ang_th, double log_eps, double density_th,
                                 int n_bins,
                                 int ** reg_img, int * reg_x, int * reg_y, double * inputv,
				 int nfunc, int ndim,
				 int nv, double * xv, double * fv, double *sigv, 
				 int ng, double* xg, double* fg, double* sigg,
				 double mirbeta, double mirgamma,int mirN, int mirP,
				 double* extrema,int prepost)
{
  image_double image;
  ntuple_list out = new_ntuple_list(7);
  double * return_value;
  image_double angles,modgrad;
  image_char used;
  image_int region = NULL;
  struct coorlist * list_p;
  void * mem_p;
  struct rect rec;
  struct point * reg;
  int reg_size,min_reg_size,i;
  unsigned int xsize,ysize;
  double reg_angle,prec,p,log_nfa,logNT;
  int ls_count = 0;                   /* line segments are numbered 1,2,3,... */


  /* check parameters */
  if( img == NULL || X <= 0 || Y <= 0 ) error("invalid image input.");
  
  if( ang_th <= 0.0 || ang_th >= 180.0 )
    error("'ang_th' value must be in the range (0,180).");
  if( density_th < 0.0 || density_th > 1.0 )
    error("'density_th' value must be in the range [0,1].");
  if( n_bins <= 0 ) error("'n_bins' value must be positive.");


  /* angle tolerance */
  prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;
 
  /* Markov kernel already computed in main pipe  */
  //make_markov( img, X, Y, ang_th, n_bins, &inputv);


double beta;
beta=inputv[0];
int sizenum;
sizenum=(int) inputv[3];
  /* load and scale image (if necessary) and compute angle at each pixel */
  image = new_image_double_ptr( (unsigned int) X, (unsigned int) Y, img );

    angles = ll_angle( image, &list_p, &mem_p, &modgrad,
                       (unsigned int) n_bins,beta);
  xsize = angles->xsize;
  ysize = angles->ysize;

  /* Number of Tests - NT

     The theoretical number of tests is Np.(XY)^(5/2)
     where X and Y are number of columns and rows of the image.
     Np corresponds to the number of angle precisions considered.
     As the procedure 'rect_improve' tests 5 times to halve the
     angle precision, and 5 more times after improving other factors,
     11 different precision values are potentially tested. Thus,
     the number of tests is
       11 * (X*Y)^(5/2)
     whose logarithm value is
       log10(11) + 5/2 * (log10(X) + log10(Y)).
  */
 
  /* initialize some structures */
  if( reg_img != NULL && reg_x != NULL && reg_y != NULL ) /* save region data */
    region = new_image_int_ini(angles->xsize,angles->ysize,0);
  used = new_image_char_ini(xsize,ysize,NOTUSED);
  reg = (struct point *) calloc( (size_t) (xsize*ysize), sizeof(struct point) );
  if( reg == NULL ) error("not enough memory!");
double p0,p1;
  p1=p;
  p0=1-p1;
  
  double p11=inputv[5];
 
  double p10=1.0-p11;
 
  
  double p01;
  p01=inputv[6];
 
  double p00;
  p00=1.0-p01;
   logNT = 5.0 * ( log10( (double) xsize ) + log10( (double) ysize ) ) / 2.0
          + log10(3.0);
   
                                   
min_reg_size=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
 
 int N=sizenum;
  double *output;
  output=( double *) malloc(N*N*sizeof(double));
  int NOUT=N;
  NFA_matrix(output,p0,p11,p01,N);
  double *output_2;
  output_2=( double *) malloc(N*N*sizeof(double));
 p11=inputv[7];
 p01=inputv[8];
p1=p/2;
p0=1-p1;
int min_reg_size_2;
min_reg_size_2=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
NFA_matrix(output_2,p0,p11,p01,N);
  double *output_4;
  output_4=( double *) malloc(N*N*sizeof(double));
  p11=inputv[9];
  p01=inputv[10];
p1=p/4;
p0=1-p1;
int min_reg_size_4;
min_reg_size_4=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
NFA_matrix(output_4,p0,p11,p01,N);

  /* search for line segments */
  for(; list_p != NULL; list_p = list_p->next )
    if( used->data[ list_p->x + list_p->y * used->xsize ] == NOTUSED &&
        angles->data[ list_p->x + list_p->y * angles->xsize ] != NOTDEF )
       /* there is no risk of double comparison problems here
          because we are only interested in the exact NOTDEF value */
      {
        /* find the region of connected point and ~equal angle */
        region_grow( list_p->x, list_p->y, angles, reg, &reg_size,
                     &reg_angle, used, prec );

        /* reject small regions */
        if( reg_size < min_reg_size ) continue;

        /* construct rectangular approximation for the region */
        region2rect(reg,reg_size,modgrad,reg_angle,prec,p,&rec);

        /* Check if the rectangle exceeds the minimal density of
           region points. If not, try to improve the region.
           The rectangle will be rejected if the final one does
           not fulfill the minimal density condition.
           This is an addition to the original LSD algorithm published in
           "LSD: A Fast Line Segment Detector with a False Detection Control"
           by R. Grompone von Gioi, J. Jakubowicz, J.M. Morel, and G. Randall.
           The original algorithm is obtained with density_th = 0.0.
         */
        if( !refine( reg, &reg_size, modgrad, reg_angle,
                     prec, p, &rec, used, angles, density_th ) ) continue;

        /* compute CONDITIONAL NFA value */

        log_nfa = rect_improve(&rec,angles,logNT,log_eps,output,output_2,output_4,NOUT,min_reg_size,min_reg_size_2,min_reg_size_4);
        
	if (prepost==0) //precondition 
	{
		/*
		   The gradient was computed with a 2x2 mask, its value corresponds to
		   points with an offset of (0.5,0.5), that should be added to output.
		   The coordinates origin is at the center of pixel (0,0).
		 */
		rec.x1 += 0.; rec.y1 += 0.;
		rec.x2 += 0.; rec.y2 += 0.;
		/* add line segment found to output */
		add_7tuple( out, rec.x1, rec.y1, rec.x2, rec.y2,
				 rec.width, rec.p, log_nfa );
		if( log_nfa <= log_eps ) continue;
		/* A New Line Segment was found! */
		++ls_count;  /* increase line segment counter */
	}
	else //postcondition 
	{
		int nx=1;
		double  *x=     (double *) malloc(1*ndim*sizeof(double));
		double  *fx=    (double *) malloc(1*1*sizeof(double));
		double  *sigma= (double *) malloc(1*1*sizeof(double));
		x[0] = (double) dist(rec.x1,rec.y1,rec.x2,rec.y2);
		x[1] = (double) line_angle(rec.x1,rec.y1,rec.x2,rec.y2);
		x[2] = (double) rec.width;
		
		//Evaluate conditioning surface for rectange
		int mircheck;
		mircheck = mirEvaluate(nfunc, ndim,  nx, x, 
			nv, xv, fv, sigv, 
			0, NULL, NULL, NULL,
			mirbeta,mirgamma,mirN,mirP,
			fx, sigma);
		if(mircheck!=0) {printf("mirerror\n"); fflush(stdout);}
		if(!isfinite((double)fx[0])) continue;
		
		//apply conditional transform 
		double fx_tol = 0.0001;
		if(fx[0]<=fx_tol) continue;
		double pfx = 1-log10(fx[0]); 
		log_nfa = log_nfa-log10(pfx);
		if(!isfinite(log_nfa)) continue;

		if( log_nfa <= log_eps ) continue;
		/* A New Line Segment was found! */
		++ls_count;  /* increase line segment counter */
		rec.x1 += 0.; rec.y1 += 0.;
		rec.x2 += 0.; rec.y2 += 0.;
		add_7tuple( out, rec.x1, rec.y1, rec.x2, rec.y2,
				 rec.width, rec.p, log_nfa );
	}
        /* add region number to 'region' image if needed */
        if( region != NULL )
          for(i=0; i<reg_size; i++)
            region->data[ reg[i].x + reg[i].y * region->xsize ] = ls_count;
      }


  /* free memory */
  free( (void *) image );   /* only the double_image structure should be freed,
                               the data pointer was provided to this functions
                              and should not be destroyed.                 */
  free_image_double(angles);
  free_image_double(modgrad);
  free_image_char(used);
  free( (void *) reg );
  free( (void *) mem_p );
free((void *) output); 
free((void *) output_2);
free((void *) output_4);
  /* return the result */
  if( reg_img != NULL && reg_x != NULL && reg_y != NULL )
    {
      if( region == NULL ) error("'region' should be a valid image.");
      *reg_img = region->data;
      if( region->xsize > (unsigned int) INT_MAX ||
          region->xsize > (unsigned int) INT_MAX )
        error("region image to big to fit in INT sizes.");
      *reg_x = (int) (region->xsize);
      *reg_y = (int) (region->ysize);

      /* free the 'region' structure.
         we cannot use the function 'free_image_int' because we need to keep
         the memory with the image data to be returned by this function. */
      free( (void *) region );
    }
  if( out->size > (unsigned int) INT_MAX )
    error("too many detections to fit in an INT.");
  *n_out = (int) (out->size);

  return_value = out->values;
  free( (void *) out );  /* only the 'ntuple_list' structure must be freed,
                            but the 'values' pointer must be keep to return
                            as a result. */

  return return_value;
}

/*----------------------------------------------------------------------------*/
/*------------------- CONDITIONING SURFACE EVALUATION ------------------------*/
/*----------------------------------------------------------------------------*/


double * surf_grid( int * n_out,
		 int nfunc, int ndim,
		 int nv, double * xv, double * fv, double *sigv, 
		 int ng, double* xg, double* fg, double* sigg,
		 double mirbeta, double mirgamma,int mirN, int mirP,
		 double* extrema_old)
{	
	//create list
	ntuple_list out = new_ntuple_list((int)(ndim+nfunc));
	double * return_value;
	int ii,jj,tol;
	double i,j,k;
	double*extrema=(double*)malloc(ndim*2*sizeof(double)); // min,max
	double*step = (double*)malloc(ndim*1*sizeof(double));
	int numsteps = 10;
	//numsteps+=2; //account for off-grid data
	int icount=0, jcount=0,kcount=0;
	int mircheck;
	int nx=1;
	double*x = (double*)malloc(ndim*1*sizeof(double));
	double*fx = (double*)malloc(1*1*sizeof(double));
	double*sigma = (double*)malloc(1*1*sizeof(double));


	double*means=(double*)malloc(ndim*1*sizeof(double));
	double*stds=(double*)malloc(ndim*1*sizeof(double));
	
	for(ii=0;ii<ndim;ii++){means[ii]=calc_mean(xv,nv,ndim,ii);}
	for(ii=0;ii<ndim;ii++){stds[ii]=calc_std(xv,nv,ndim,ii);}
	
	//add original points
	for (ii=0;ii<nv;ii++){add_4tuple(out,xv[ii*ndim+0],xv[ii*ndim+1],xv[ii*ndim+2],fv[ii]);}
	//find grid limits
	for (tol=1;tol<4;tol++)
	{
		for (ii=0;ii<ndim;ii++){extrema[2*ii+0]=180;extrema[2*ii+1]=-180;}
		for (ii=0;ii<nv;ii++)
		{
			for(jj=0;jj<ndim;jj++)
			{
				if(extrema[2*jj+0]>xv[ii*ndim+jj]) extrema[2*jj+0]=means[jj]-1*(double)tol*stds[jj];
					//extrema[2*jj+0]=xv[ii*ndim+jj];
				if(extrema[2*jj+1]<xv[ii*ndim+jj])  extrema[2*jj+1]= means[jj]+(double)tol*stds[jj];
					//extrema[2*jj+1]=xv[ii*ndim+jj];
			}
		}	
		
		
		
		//get grid sizing
		printf("length - min: %.2f, max: %.2f\n",extrema[0],extrema[1]);
		printf("orient - min: %.2f, max: %.2f\n",extrema[2],extrema[3]);
		printf("width  - min: %.2f, max: %.2f\n",extrema[4],extrema[5]);
		fflush(stdout);
		
		for(ii=0;ii<ndim;ii++){step[ii] = (extrema[2*ii+1]-extrema[2*ii+0])/numsteps;}
		//interpolate over grid (with 1-step exceedence);
		//
		for(i=extrema[2*0+0]-step[0];i<=extrema[2*0+1]+step[0];i+=step[0])
		{
			printf("IGrid iteration: %d of %d\n",icount,numsteps);fflush(stdout);icount++;
			for(j=extrema[2*1+0]-step[1];j<=extrema[2*1+1]+step[1];j+=step[1])
			{
				
				//printf("\tJGrid iteration: %d of %d\n",jcount,numsteps);
				//printf("\t min: %.4f, max: %.4f, step: %.4f,\n",extrema[2*2+0],extrema[2*2+1],step[2]);
				
				//fflush(stdout);jcount++;
				for(k=extrema[2*2+0]-step[2];k<=extrema[2*2+1]+step[2];k+=step[2])
				{
				
					//printf("\t\tKGrid iteration: %d of %d\n",kcount,numsteps);
					//printf("\t\t\t val: %.4f\n",k);

					//fflush(stdout);kcount++;
					x[0]=i;x[1]=j;x[2]=k;
					mircheck = mirEvaluate(nfunc, ndim,  nx, x, 
						nv, xv, fv, sigv, 
						0, NULL, NULL, NULL,
						mirbeta,mirgamma,mirN,mirP,
						fx, sigma);
					if(mircheck!=0) {printf("mirerror\n"); fflush(stdout);}
					if(!isfinite((double)fx[0])) continue;
					add_4tuple(out,i,j,k,fx[0]);
				}
				kcount=0;
			}
			jcount=0;
		}
	}
	//
	//return statements
  	if( out->size > (unsigned int) INT_MAX )
    		error("too many detections to fit in an INT.");
  	*n_out = (int) (out->size);
	return_value=out->values;
	free((void*)out);
	return return_value;
}

/*----------------------------------------------------------------------------*/
/*------------------------- OBSERVATION SAMPLING -----------------------------*/
/*----------------------------------------------------------------------------*/

//sorting structure 
struct sortstr
{
	double value;
	int index;
};

//ascending sorting algorithm 
int cmp(const void *a, const void *b)
{
    //+1,-1 for ascending
    //-1,+1 for descending 
    struct sortstr *a1 = (struct sortstr *)a;
    struct sortstr *a2 = (struct sortstr *)b;
    if ((*a1).value > (*a2).value) return 1;
    else if ((*a1).value < (*a2).value)  return -1;
    else return 0;
}

/*--------------------------------------------------------------------------*/
/* Sort (nfa,rec) data points by clustering rec grounds around nfa minima,
 * for faster automatic fitting of interpolation parameters
 * */
int * sort_list(double * fv, double * xv, int nv, int ndim,double tol,int N)
{
	printf("nv : %d, N : %d\n",nv,N);fflush(stdout);
	if(N>=nv) error("In sort_list clustering, must have N<nv.\n");
	int *sort_list=(int*) malloc(nv*1*sizeof(int));
	int sort_count = 0, unique_count=0;
	int f0=0;
	int  *match= (int *) malloc(nv*1*sizeof(int));
	int i,j,param,count;
	double paramtol;
	while (sort_count<N){
		//get clustering center point, and fresh match list
		//sort_list[sort_count]=f0;sort_count++;
		for(i=0;i<(f0-1);i++){match[i]=0;}
		for(i=f0;i<nv;i++){match[i]=1;}
		//get matchlist for point
		for(param=0;param<(ndim-1);param++)	{
			//get parameter std
			paramtol=calc_std(xv,nv,ndim,param)*tol;
			//for each parameter, update the match list if fails test
			for(i=f0;i<nv;i++){//f0 will be added by default
				if(abs(xv[f0*ndim+param]-xv[i*ndim+param])>=paramtol){
				       	match[i]=0;
				}
			}
		}
		//assign matches to list
		for(i=f0;i<nv;i++){
			if(match[i]==1){sort_list[sort_count]=i;sort_count++;}
				//printf("match %.4f\n",fv[i]);fflush(stdout);}
		}
		f0++; 
		if(f0>=N) break; //termination for safety 	
	}
	//cut unique 
	int* sort_list_unique = (int *)malloc(sort_count*1*sizeof(int));
	printf("SORT COUNT: %d\n",sort_count);fflush(stdout);
	for(i=0;i<sort_count;i++){
		count=0;
		//if an index exists later in the list, skip it
		//the index will be uniquely included in its last occurance
		for(j=i+1;j<sort_count;j++){if(sort_list[i]==sort_list[j]) count++;}
		if(count==0){
			sort_list_unique[unique_count]=sort_list[i];
			unique_count++;
		}
	}
	printf("UNIQUE COUNT: %d\n",unique_count);fflush(stdout);
	//output is size N, buffered with nulls if unique<N
	int* sort_list_final = (int *)malloc(N*1*sizeof(int));
	if(unique_count>N) unique_count=N;
	else{for(i=unique_count+1;i<N;i++){sort_list_final[i]=-1;}}//NULL;}}
	for(i=0;i<unique_count;i++){sort_list_final[i]=sort_list_unique[i];
		//printf("list %d: %d\n",i,sort_list_final[i]);fflush(stdout);
	}
	free((void*)sort_list);
	free((void*)match);
	free((void*)sort_list_unique);
	return sort_list_final;
	//size = sizeof(sort_list_unique)/sizeof(int)
	//returns a list of indices for the desired set
}



/*----------------------------------------------------------------------------*/
/*-------------------------- Line Segment Detector ---------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/* For an noise and observation image pair, 
 * i)   estimate the Markov transition kernel, 
 * ii)  capture a nfa conditioning table using the noise image,
 * iii) estimate interpolator parameters on a clustered subsample of conditioning points,
 * iv)  and run the Conditional LSD on the observation using the conditioning surface. 
 * */
double * CLSD_Pipeline( int * n_out,
			   double * img0, int X0, int Y0,
			   double * img, int X, int Y,
			   double ang_th, double log_eps, double density_th, int n_bins, 
                           int ** reg_img, int * reg_x, int * reg_y, double * inputv)
{
	int i,j,il;
	int mm=7;

	//Fix sizenum variable for noise image 
	double sizenum = sqrt(pow((double)X0,2)+pow((double)Y0,2))*5.;
	if(sizenum>pow(10.,4)) sizenum=pow(10.,4);
	inputv[3] = sizenum;

	//Caluculate Markov kernel
	make_markov( img0, X0, Y0, ang_th, n_bins, inputv);

	//Capture conditioning table 
	int n_out_I0;
	double * out_I0 = Conditional_LineSegmentDetection( &n_out_I0, img0,X0,Y0,
                               ang_th, log_eps, density_th, n_bins,
                               NULL, NULL, NULL ,inputv,
			       0,0,0,NULL,NULL,NULL,
			       0,NULL,NULL,NULL,
			       0, 0, 0, 0, NULL,
			       0);//last term informs post conditionin pattern 
	printf("Precondition points:%d\n",n_out_I0); fflush(stdout);
    
	/* Instatntiate the interpolant.  
	 * Given fv = f(xv), Solve f(x)
	 */
	int nfunc=1;          // number functions to approximate
	int ndim=3;           // dimensions of approximation space
	int nv=n_out_I0;      // number of value data points to approximate 
	double  *xv= (double *) malloc(ndim*nv*sizeof(double)); //rectangle parameters
	double *fv = (double *) malloc(nv*nfunc*sizeof(double));//nfa for rectangels
	double *sigv = (double *) malloc(nv*1*sizeof(double));  //uncertainty 
	//assuming no gradient knowlege  (ng, xg, fg, sigg) 
	double beta; // magnitude parameter
	double gamma;// wavenumber parameter
	int N;       // Taylor order parameter
	int P;       // Polynomial exactness parameter
	
	// Store rectangle data into interpolant table 
	// Fill out xv=(lenth,orientation,width), fv=(NFA) data 
	// fv is normalized during computation 
	double fv_sum=0, fv_max=0, fv_min=10;
	for(i=0;i<nv;i++)
	{
		xv[i*ndim+0] = dist(out_I0[i*mm+j+0],out_I0[i*mm+j+1],
						out_I0[i*mm+j+2],out_I0[i*mm+j+3]);
		xv[i*ndim+1] = line_angle(out_I0[i*mm+j+0],out_I0[i*mm+j+1],
						out_I0[i*mm+j+2],out_I0[i*mm+j+3]);
		xv[i*ndim+2] = out_I0[i*mm+j+4];
		//xv[i*ndim+3] = out_I0[i*mm+j+5];
		fv[i] = out_I0[i*mm+j+6];
		if(!isfinite((double)fv[i])) fv[i] = 0;
		sigv[i]=0;
	}
	//Set fv = min(0,log10(eps/NFA)) and normalize max(fv)=1 
	for(i=0;i<nv;i++)
	{
		if(fv[i]>log_eps) fv[i] = 0.0;
		else
		{
			fv[i] = log_eps - fv[i]; //log(e/NFA)
			//Collect statistical data
			fv_sum+=fv[i];
			if(fv[i]>fv_max) fv_max=fv[i];
			if(fv[i]<fv_min) fv_min=fv[i];
			if(!isfinite(fv[i])) fv[i]=1.0;
		}
	}
	for(i=0;i<nv;i++) {fv[i]/=fv_max;}
	printf("fvmax : %.4f, fvmin : %.4f, fvmean: %.4f\n",fv_max,fv_min,fv_sum/nv);	
	fflush(stdout);
	
	//Get new fv statistics 
	double fx_tol = 0.0001;
	double fv_0card=0;
	fv_sum=0;
	for(i=0;i<nv;i++)
	{
		fv_sum+=fv[i];
		if(fv[i]<fx_tol) fv_0card++;
	}
	
	double card_factor = fv_0card/nv;
	double fv_mean = fv_sum/nv;
	double fv_std=0;
	for(i=0;i<nv;i++){fv_std+=pow(fv[i]-fv_mean,2);}
	fv_std= sqrt(fv_std/nv);	
	printf("fvmean: %.4f, fvstd: %.4f\n",fv_mean, fv_std);	
	printf("0cardinality: %.1f, fractional: %.4f\n",fv_0card,card_factor);
	
	// INITIAL ASCENDING SORT
	struct sortstr fvsort[nv];
	for (i=0;i<nv;i++){fvsort[i].value=fv[i]; fvsort[i].index=i;}
	int nv_sorted = nv;      // number of value data points to approximate 
	qsort(fvsort,nv,sizeof(fvsort[0]),cmp);
	
	double  *xv_sorted= (double *) malloc(ndim*nv_sorted*sizeof(double));
	double *fv_sorted = (double *) malloc(nv_sorted*nfunc*sizeof(double)); 
	double *sigv_sorted = (double *) malloc(nv_sorted*1*sizeof(double));   
	printf("Sorting all points...\n");fflush(stdout);
	for (i=0;i<nv_sorted;i++)
	{
		il = (int) fvsort[i].index;
		for (j=0;j<ndim;j++) xv_sorted[ndim*i+j] = xv[ndim*il+j];
		fv_sorted[i] = fv[il];
		sigv_sorted[i] = sigv[il];
	}
	
	// CLUSTER SORT BY XV AROUND FV MINIMA
	int nv_sorted_2=150; //target nv length, from which we get unique.
	double sort_tolerance_factor = 1.0; //sigma-multiple for tolerance	
	printf("Clustering points...\n");fflush(stdout);
	int * sort_idxs= sort_list(fv_sorted, xv_sorted, nv_sorted, ndim, 
			sort_tolerance_factor, nv_sorted_2);
	int nv_sorted_2_corrected=0;
	for(i=0;i<nv_sorted_2;i++){
		if(sort_idxs[i]==-1) break;//NULL) break;	
		nv_sorted_2_corrected++;
	}
	printf("RETURNED SIZE: %d\n",nv_sorted_2_corrected);fflush(stdout);
	double  *xv_sorted_2= (double *) malloc(ndim*nv_sorted_2*sizeof(double));
	double *fv_sorted_2 = (double *) malloc(nv_sorted_2*nfunc*sizeof(double));
	double *sigv_sorted_2 = (double *) malloc(nv_sorted_2*1*sizeof(double));  
	for (i=0;i<nv_sorted_2_corrected;i++)
	{
		il = (int) sort_idxs[i];
		for (j=0;j<ndim;j++) xv_sorted_2[ndim*i+j] = xv_sorted[ndim*il+j];
		fv_sorted_2[i] = fv_sorted[il];
		sigv_sorted_2[i] = sigv_sorted[il];
	}
	

	//RUN PARAMETER ESTIMATION 
	int    mircheck;
	double mirbeta  = 0.02;
	double mirgamma = 7.55;
	int    mirP     = 1;
	int mirN=10;
	if(card_factor==0) error("MirBetaGamma: Cannot compute: cardinaltiy factor == 0");
	double mirsafety =1./(2.*card_factor); //higher than 1 for  lower-order approx 
	printf("SAFETY: %.2f\n",mirsafety);fflush(stdout);
	int bgcheck=0;
	bgcheck = mirBetaGamma(nfunc,ndim,
			nv_sorted_2,xv_sorted_2,fv_sorted_2,sigv_sorted_2,
			0,NULL,NULL,NULL,
			mirN,mirP,mirsafety,
			&mirbeta,&mirgamma);
	if(bgcheck!=0) {error("mirbetagamma faile");printf("bgerror\n"); fflush(stdout);}
	if(!isfinite(mirgamma)) {error("mirgamma non-finite");printf("bgerror\n"); fflush(stdout);}
	printf("Estimated mir params:: beta=%.2f, gamma=%.2f, N=%d, P=%d\n",
			mirbeta,mirgamma,mirN,mirP);
	fflush(stdout);
	
	//Update Sizenum for observation image
	sizenum = sqrt(pow((double)X,2)+pow((double)Y,2))*5.;
	if(sizenum>pow(10.,4)) sizenum=pow(10.,4);
	inputv[3] = sizenum;

	//Run either CLSD or aquire conditioning surface for plotting 
	double  *extrema= (double *) malloc(ndim*2*sizeof(double));
	if (X==0) {return surf_grid(n_out,
			       nfunc, ndim, nv, xv, fv, sigv, 
			       0, NULL, NULL, NULL,
			       mirbeta,mirgamma,mirN,mirP, extrema);}
	else {	return Conditional_LineSegmentDetection(n_out, img,X,Y,
                               ang_th, log_eps, density_th, n_bins,
                               NULL, NULL, NULL ,inputv,
			       nfunc, ndim, nv, xv, fv, sigv, 
			       0, NULL, NULL, NULL,
			       mirbeta,mirgamma,mirN,mirP, extrema,
			       1);}
}


/*----------------------------------------------------------------------------*/
/*----------------------- LSD-WRAPPER INTERFACES -----------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** LSD Simple Interface with Scale and Region output.
 */
double * lsd_scale_region( int * n_out,
                           double * img, int X, int Y,
                           int ** reg_img, int * reg_x, int * reg_y,double * inputv )
{
  /* LSD parameters */
  
  double ang_th;   /* Gradient angle tolerance in degrees.           */
  ang_th=inputv[4];
  double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  log_eps=-log10(inputv[1]);
  
  double density_th = 0.0;  /* Minimal density of region points in rectangle. */
  density_th=inputv[2];
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */

  return LineSegmentDetection( n_out, img, X, Y,
                               ang_th, log_eps, density_th, n_bins,
                               reg_img, reg_x, reg_y ,inputv);
}

/*----------------------------------------------------------------------------*/
/** LSD Simple Interface with Scale.
 */
double * lsd_scale(int * n_out, double * img, int X, int Y,double * inputv)
{
  return lsd_scale_region(n_out,img,X,Y,NULL,NULL,NULL,inputv);
}

/*----------------------------------------------------------------------------*/
/** LSD Simple Interface.
 */
double * lsd(int * n_out, double * img, int X, int Y,double *inputv)
{
  /*fprintf(stderr,"LSD Error: %s\n","Just Testing");*/
  return lsd_scale(n_out,img,X,Y,inputv);
}

/*----------------------------------------------------------------------------*/
/** CLSD Simple Interface with Scale and Region output.
 */
double * c_lsd_scale_region( int * n_out,
                             double * img, int X, int Y,
			     double * img0, int X0, int Y0,
                             int ** reg_img, int * reg_x, int * reg_y,double * inputv )
{
  /* LSD parameters */
  
  double ang_th;   /* Gradient angle tolerance in degrees.           */
  ang_th=inputv[4];
  double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  log_eps=-log10(inputv[1]);
  
  double density_th = 0.0;  /* Minimal density of region points in rectangle. */
  density_th=inputv[2];
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */
  return CLSD_Pipeline(n_out,
		   	img0, X0, Y0,
		   	img, X, Y,
		   	ang_th, log_eps, density_th, n_bins,
                   	reg_img, reg_x, reg_y ,inputv);
}

/*----------------------------------------------------------------------------*/
/** CLSD Simple Interface with Scale.
 */
double * c_lsd_scale(int * n_out, 
				     double * img, int X, int Y,
				     double * img0, int X0, int Y0, 
				     double * inputv)
{
  return c_lsd_scale_region(n_out,img,X,Y,img0,X0,Y0,NULL,NULL,NULL,inputv);
}

/*----------------------------------------------------------------------------*/
/** CLSD Simple Interface.
 */
double * c_lsd(int * n_out, 
			   double * img, int X, int Y,
			   double * img0, int X0, int Y0,
			   double *inputv)
{

  return c_lsd_scale(n_out,img,X,Y,img0,X0,Y0,inputv);
}

/*----------------------------------------------------------------------------*/
/*------------------------------ PYTHON WRAPPER ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/

/* Main Python interface.
 * Converts Python/C memory structures and decides between clsd/lsdsar pipes.
 * Usage: from clds import clsd; 
 *        inputv=[alpha,eps,density,sizenum,angth,p11,p01,p11_2,p01_2,p11_4,p01_4]
 *        lines = clds(I,X,Y,I0,X0,Y0,inputv) 
 * Input: I - flattened X-by-Y array of observation image data
 *        I0- flattened X0-by-Y0 array of noise model image data
 *        inputv - (5+6)-length vector of pipeline parameters, accourding to LSDSAR
 *           alpha   - weighting parameter for Grdient-by-Ratio calulation, chosen as
 *                      4 by empirical study on Nakagami noise.
 *           eps     - NFA threshold, set to 1 by a contrario theory  
 *           density - threshold for refining bounding rectangles to a specified density, 
 *                       chosen as 0.4 default for avoiding 'nested' detections. 
 *                       Note: this may need increased to prevent over-cutting, or
 *                       lowered to prevent inconsistencies in detection.  
 *           sizenum - upper bound for maximum area of bounding rectangles to consider
 *                       before automatric acceptance.  Choose max[10^4,(X^2+Y^2)^(5/2)].
 *                       Note: Chosen internally for processing both I and I0; ignore. 
 *           angth   - absolute angular threshold for 'alignment' of points.  Chosen as
 *                       (pi/8)rads, i.e. 1/8 the full randge of orientation
 *           px1     - Markov transition probabilities for 1, 1/2, and 1/4 angth.
 *                       May be set to 0 if I0 is present for automatic estimation 
 * Output: lines - a N-by-7 list of properties for N lines, being 
 *                 (x1,y1,x2,y2,width,angleth,nfa) as long as X>0
 *                 If X==0, i.e. only the conditional image, 
 *                 this is a (length,orientation,width,nfa) N-by-4 vector, used
 *                 for mapping the conditional space.  
 *                 If X0==0, i.e. no conditional image,
 *                 the input Markov transition kernel is used, and conditioning
 *                 is avoided (running the original LSDSAR algorithm).  
 *
 */
 
static PyObject * clsd(PyObject * self, PyObject * args)
{

	
	//instantiate intermediate pointers 
	PyObject * imagein;
	PyObject * image0in;
	PyObject * inputvin;
	double * image;
	double * image0;
	double * inputv;
	double * out;
	double * outdata;
	int n_points;
	int x,y,i,j,n;
	int X,Y; 
	int X0,Y0;
	int Xin,Yin;

  	// parse arguments
  	if (!PyArg_ParseTuple(args, "OIIOIIO", &imagein, &X, &Y, &image0in, &X0, &Y0, &inputvin)) {
    		return NULL;
  	} 
	imagein = PySequence_Fast(imagein, "arguments must be iterable");
	if(!imagein){
		return 0;
	}
	image0in = PySequence_Fast(image0in, "arguments must be iterable");
	if(!image0in) {
		return 0;
	}
	inputvin = PySequence_Fast(inputvin, "arguments must be iterable");
	if(!inputvin) {
		return 0;
	}
	
	// pass Python data to C structures
	n_points = PySequence_Fast_GET_SIZE(imagein);
	image = malloc(n_points*sizeof(double));
        if(!image){
		//Py_DECREF(imagein);
		return PyErr_NoMemory( );
	}	
	for (i=0; i<n_points; i++) {
		PyObject *fitem;
		PyObject *item = PySequence_Fast_GET_ITEM(imagein, i);
		if(!item) {
			//Py_DECREF(imagein);
			free(image);
			return 0;
		}
		fitem = PyNumber_Float(item);
		if(!fitem) {
			//Py_DECREF(imagein);
			free(image);
			PyErr_SetString(PyExc_TypeError, "all items must be numbers");
			return 0;
		}
		image[i] = PyFloat_AS_DOUBLE(fitem);
		Py_DECREF(fitem);
	}
	
	n_points = PySequence_Fast_GET_SIZE(image0in);
	image0 = malloc(n_points*sizeof(double));
        if(!image0){
		//Py_DECREF(image0in);
		return PyErr_NoMemory( );
	}	
	for (i=0; i<n_points; i++) {
		PyObject *fitem;
		PyObject *item = PySequence_Fast_GET_ITEM(image0in, i);
		if(!item) {
			//Py_DECREF(image0in);
			free(image0);
			return 0;
		}
		fitem = PyNumber_Float(item);
		if(!fitem) {
			//Py_DECREF(image0in);
			free(image0);
			PyErr_SetString(PyExc_TypeError, "all items must be numbers");
			return 0;
		}
		image0[i] = PyFloat_AS_DOUBLE(fitem);
		Py_DECREF(fitem);
	}
	
	n_points = PySequence_Fast_GET_SIZE(inputvin);
	inputv = malloc(n_points*sizeof(double));
        if(!inputv){
		//Py_DECREF(inputvin);
		return PyErr_NoMemory( );
	}	
	for (i=0; i<n_points; i++) {
		PyObject *fitem;
		PyObject *item = PySequence_Fast_GET_ITEM(inputvin, i);
		if(!item) {
			//Py_DECREF(inputvin);
			free(inputv);
			return 0;
		}
		fitem = PyNumber_Float(item);
		if(!fitem) {
			//Py_DECREF(inputvin);
			free(inputv);
			PyErr_SetString(PyExc_TypeError, "all items must be numbers");
			return 0;
		}
		inputv[i] = PyFloat_AS_DOUBLE(fitem);
		Py_DECREF(fitem);
	}	

  	// Pick and run the CLSD or LSDSAR algorithm
  	if((X0==0) & (X>0))
	{
		printf("Using existing Markov kernel\n");
		fflush(stdout);
		out = lsd(&n,image,X,Y,inputv);
	}
	else
	{
	  	printf("Estimating Markov kernel\n");
	  	printf("old: (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f) \n",inputv[5],inputv[6], inputv[7],inputv[8],inputv[9],inputv[10]);
	  	out = c_lsd(&n,image,X,Y,image0,X0,Y0,inputv);
	  	printf("new: (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f)\n",inputv[5],inputv[6], inputv[7],inputv[8],inputv[9],inputv[10]);
	  	fflush(stdout);
	}
	free(image);
	free(image0);
	free(inputv);
	
	// Convert output to a valid Python structure
	// accounting for various dimensionality options
	int mm=7;
	if(X==0) mm=4;
	PyObject * pyout = PyList_New((int)n*mm);
	if (!pyout) {
		return NULL;
	}	
	for (i = 0; i< n; i++) 
	{
		for(j=0;j<mm;j++) 
		{
			PyObject *num = PyFloat_FromDouble(out[i*mm+j]);
			if (!num) {
				Py_DECREF(pyout);
				return NULL;
			}
			PyList_SET_ITEM(pyout, i+j*n, num);
		}
	}
  	return pyout; 
}

//Pythonic interfaces
static PyMethodDef clsdMethods[] = {
	{"clsd", clsd, METH_VARARGS, "Conditional variation of the LSDSAR algorithm"},
	{NULL, NULL, 0, NULL} /* sentinel */
};

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"clsd",
	NULL,
	-1,
	clsdMethods
};

PyMODINIT_FUNC PyInit_clsd(void)
{
	return PyModule_Create(&moduledef);
}
