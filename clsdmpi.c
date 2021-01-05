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
//#define _GNU_SOURCE
#include <Python.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include<string.h>
#include <time.h>
#include<gsl/gsl_eigen.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_qrng.h>
#include<gsl/gsl_sf_trig.h>
#include<mir/mir.h>
#include<sys/mman.h>
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
struct coorlist3
{
    int x,y,z;
    struct coorlist3 * next;
};
/*----------------------------------------------------------------------------*/
/** A point (or pixel).
 */
struct point {int x,y;};
struct point3 {int x,y,z;};

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

static double dist3(double x1, double y1, double z1, double x2, double y2, double z2)
{
  return sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1) );
}

/*----------------------------------------------------------------------------*/
/* Orientation of a line*/
static double line_angle(double x1, double y1, double x2, double y2)
{
  return atan2( y2-y1 , x2-x1 );
}

/*[azimuth,elevation,quaterion] structure
 * For loop structure, one of these may have its az/el data incrementally changed,
 * and fed to calc_quat below*/
typedef struct angles3_s
{
    double az,el;  
    double quat[4];
} * angles3;

/*Given an pointer to a angles3 structure with az/el data, compute its quaternion*/
static void calc_quat(angles3 * angles)
{

    //double * quat = (double*) malloc(4*sizeof(double));
    
    //printf("\t\t\t\t\t Access angles...\n");
    fflush(stdout);
    /*
    double quat[4];
    double az = (double) (*angles) -> az;
    double el = (double) (*angles) -> el;
    //azimuth==yaw, pitch==elevation, roll==0
    
    //printf("\t\t\t\t\t Calculate...\n");
    //fflush(stdout);
    double cy = cos(az/2.);
    double sy = sin(az/2.);
    double cp = cos(el/2.);
    double sp = sin(el/2.);
    double cr = 1;//cos(0);
    double sr = 0;//sin(0);
    quat[0] = cr * cp * cy + sr * sp * sy;
        quat[1] = sr * cp * cy - cr * sp * sy;
        quat[2] = cr * sp * cy + sr * cp * sy;
        quat[3] = cr * cp * sy - sr * sp * cy;

    //printf("\t\t\t\t\t Storage...\n");
    //fflush(stdout);
    for(int i=0;i<4;i++)
        (*angles)->quat[i] = (double)quat[i];
    */
}


/** Create a new image_double of size 'xsize' times 'ysize'.
 */
static angles3 new_angles3(double az, double el)
{
  angles3 image;

  /* get memory */
  image = (angles3) malloc( sizeof(struct angles3_s) );
  if( image == NULL ) error("not enough memory.");


  image->az = az;
  image->el = el; 
  calc_quat(&image);
  return image;
}

/*----------------------------------------------------------------------------*/
/** Free memory used in image_double 'i'.
 */
static void free_angles3(angles3 i)
{
  if( i == NULL)
    error("free_angles3: invalid input image.");
  free( (void *) i );
}
/*Given line endpoints, return a structure [azimuth, elevation, quaternion] */
static angles3 line_angle3(double x1, double y1, double z1, 
        double x2, double y2, double z2)
{
    double az =  atan2( y2-y1 , x2-x1) ;
    double el =  acos((z2-z1)/dist3(x1,y1,z1,x2,y2,z2));
    angles3 azel=new_angles3(az,el);
    //azel->az = az;
    //azel->el = el;
    //calc_quat(&azel);
    return azel;
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
/** Unsigned angle difference, -pi/2,pi/2, orthogonality at 0
 */
static double quat_diff(angles3 a, angles3 b)
{
  //printf("\t\t\t\t\t Compute quaternions: az %.2f, el %.2f, quat [%.2f,%.2f,%.2f,%.2f]...\n",a->az,a->el,a->quat[0],a->quat[1],a->quat[2],a->quat[3]);
  fflush(stdout);
  if(a->quat==NULL) calc_quat(&a);
  if(b->quat==NULL) calc_quat(&b);
  //calc_quat(&a);
  //calc_quat(&b);
  double dot=0.;


  double  dl1[3] =  {cos(a->az)*sin(a->el), sin(a->az)*sin(a->el),cos(a->el)};
  double  dl2[3] =  {cos(b->az)*sin(b->el), sin(b->az)*sin(b->el),cos(b->el)};

  for(int i=0;i<3;i++) dot+=dl1[i]*dl2[i];
  //dot = sin(a->el)*sin(b->el)*cos(a->az - b->az) + cos(a->el)*cos(b->el);
  //printf("\t\t\t\t\t Compute metric...\n");
  //fflush(stdout);
  //for(int i=0;i<4;i++) dot+=(a->quat[i])*(b->quat[i]);
  
  //printf("\t\t\t\t\t %.2f\n",fabs(dot));
  //fflush(stdout);
  return fabs(dot);
}

static double quat_diff_signed(angles3 a, angles3 b)
{

  fflush(stdout);
  if(a->quat==NULL) calc_quat(&a);
  if(b->quat==NULL) calc_quat(&b);

  double dot=0.;
  double  dl1[3] =  {cos(a->az)*sin(a->el), sin(a->az)*sin(a->el),cos(a->el)};
  double  dl2[3] =  {cos(b->az)*sin(b->el), sin(b->az)*sin(b->el),cos(b->el)};

  for(int i=0;i<3;i++) dot+=dl1[i]*dl2[i];

  //for(int i=0;i<4;i++) dot+=(a->quat[i])*(b->quat[i]);
  return dot;
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

static void add_10tuple( ntuple_list out, double v1, double v2, double v3,
                        double v4, double v5, double v6, double v7,
                double v8, double v9, double v10)
{
  /* check parameters */
  if( out == NULL ) error("add_10tuple: invalid n-tuple input.");
  if( out->dim != 10 ) error("add_10tuple: the n-tuple must be a 10-tuple.");

  /* if needed, alloc more tuples to 'out' */
  if( out->size == out->max_size ) enlarge_ntuple_list(out);
  if( out->values == NULL ) error("add_10tuple: invalid n-tuple input.");

  /* add new 7-tuple */
  out->values[ out->size * out->dim + 0 ] = v1;
  out->values[ out->size * out->dim + 1 ] = v2;
  out->values[ out->size * out->dim + 2 ] = v3;
  out->values[ out->size * out->dim + 3 ] = v4;
  out->values[ out->size * out->dim + 4 ] = v5;
  out->values[ out->size * out->dim + 5 ] = v6;
  out->values[ out->size * out->dim + 6 ] = v7;
  out->values[ out->size * out->dim + 7 ] = v8;
  out->values[ out->size * out->dim + 8 ] = v9;
  out->values[ out->size * out->dim + 9 ] = v10;
  
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
/*-------------------------- 3D Image Data Types -----------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** char image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image3_char_s
{
  unsigned char * data;
  unsigned int xsize,ysize,zsize;
} * image3_char;

/*----------------------------------------------------------------------------*/
/** Free memory used in image_char 'i'.
 */
static void free_image3_char(image3_char i)
{
  if( i == NULL || i->data == NULL )
    error("free_image3_char: invalid input image.");
  free( (void *) i->data );
  free( (void *) i );
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize'.
 */
static image3_char new_image3_char(unsigned int xsize, unsigned int ysize, unsigned int zsize)
{
  image3_char image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 || zsize == 0) error("new_image3_char: invalid image size.");

  /* get memory */
  image = (image3_char) malloc( sizeof(struct image3_char_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (unsigned char *) calloc( (size_t) (xsize*ysize*zsize),
                                          sizeof(unsigned char) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;
  image->zsize = zsize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
static image3_char new_image3_char_ini( unsigned int xsize, unsigned int ysize, unsigned int zsize,
                                      unsigned char fill_value )
{
  image3_char image = new_image3_char(xsize,ysize,zsize); /* create image */
  unsigned int N = xsize*ysize*zsize;
  unsigned int i;

  /* check parameters */
  if( image == NULL || image->data == NULL )
    error("new_image3_char_ini: invalid image.");

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
typedef struct image3_int_s
{
  int * data;
  unsigned int xsize,ysize,zsize;
} * image3_int;

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize'.
 */
static image3_int new_image3_int(unsigned int xsize, unsigned int ysize, unsigned int zsize)
{
  image3_int image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 || zsize == 0 ) error("new_image3_int: invalid image size.");

  /* get memory */
  image = (image3_int) malloc( sizeof(struct image3_int_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (int *) calloc( (size_t) (xsize*ysize*zsize), sizeof(int) );
  if( image->data == NULL ) error("not enough memory.");

  /* set imagsavelines=zeros(size(lines,1),5);
for xx=1:size(lines,1)
    savelines(xx,1:4)=lines(xx,1:4);
    savelines(xx,5)=angleline(xx);
ende size */
  image->xsize = xsize;
  image->ysize = ysize;
  image->zsize = zsize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
static image3_int new_image3_int_ini( unsigned int xsize, unsigned int ysize, unsigned int zsize,
                                    int fill_value )
{
  image3_int image = new_image3_int(xsize,ysize,zsize); /* create image */
  unsigned int N = xsize*ysize*zsize;
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
typedef struct image3_double_s
{
  double * data;
  unsigned int xsize,ysize,zsize;
} * image3_double;

/*----------------------------------------------------------------------------*/
/** Free memory used in image_double 'i'.
 */
static void free_image3_double(image3_double i)
{
  if( i == NULL || i->data == NULL )
    error("free_image3_double: invalid input image.");
  free( (void *) i->data );
  free( (void *) i );
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'.
 */
static image3_double new_image3_double(unsigned int xsize, unsigned int ysize, unsigned int zsize)
{
  image3_double image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 || zsize == 0) error("new_image3_double: invalid image size.");

  /* get memory */
  image = (image3_double) malloc( sizeof(struct image3_double_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (double *) calloc( (size_t) (xsize*ysize*zsize), sizeof(double) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;
  image->zsize = zsize;
  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'
    with the data pointed by 'data'.
 */
static image3_double new_image3_double_ptr( unsigned int xsize,
                                          unsigned int ysize, unsigned int zsize,  double * data )
{
  image3_double image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 || zsize == 0 )
    error("new_image3_double_ptr: invalid image size.");
  if( data == NULL ) error("new_image3_double_ptr: NULL data pointer.");

  /* get memory */
  image = (image3_double) malloc( sizeof(struct image3_double_s) );
  if( image == NULL ) error("not enough memory.");

  /* set image */
  image->xsize = xsize;
  image->ysize = ysize;
  image->zsize = zsize;
  
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
    before the sub-sampling to prevent cosasing.

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
/*
 *3D Gradient, output [modgrad, ang1, ang2].  
 *Should be a void function and take empty ang1 ang2 images for output.
 * */



/*Access as grads.azgrad, grads.elgrad,
 * instatiate as struct grads newgrad*/
typedef struct grads_s {
    image3_double az;
        image3_double el;
} * grads;

/** Create a new image_double of size 'xsize' times 'ysize'.
 */
static grads new_grads(unsigned int xsize, unsigned int ysize, unsigned int zsize)
{
  grads image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 || zsize == 0) error("new_image3_double: invalid image size.");

  /* get memory */
  image = (grads) malloc( sizeof(struct grads_s) );
  if( image == NULL ) error("not enough memory.");



  image->az = new_image3_double(xsize,ysize,zsize);
  image->el = new_image3_double(xsize,ysize,zsize); 
      //(double *) calloc( (size_t) (xsize*ysize*zsize), sizeof(double) );
  if( image->az == NULL || image->az->data == NULL ) error("not enough memory.");
  if( image->el == NULL || image->el->data == NULL ) error("not enough memory.");
  /* set image size */
  //image->az->xsize = xsize;
  //image->az->ysize = ysize;
  //image->az->zsize = zsize;
  return image;
}

/*----------------------------------------------------------------------------*/
/** Free memory used in image_double 'i'.
 */
static void free_grads(grads i)
{
  if( i == NULL || i->az->data == NULL )
    error("free_grads: invalid input image.");
  free_image3_double(i->az );
  free_image3_double(i->el );
  free( (void *) i );
}


/*----------------------------------------------------------------------------*/
/** Computes the direction of the GRADIENT VECTOR of 'in' at each point.
 * Since the 'Level Line' is an ambiguous 2D surface in 3D.
 * The GR algorithm is identical to 2D, but we switch the atan expression 
 * to use the normal and not tangential direction
 * This correction hence mandates the following changes:
 *
 * get_theta3: keeps min eigenvalue for principal axis
 * 	*note: called by region2rect3 to assemble rectangle
 * region3_grow: keeps parallel alignment of neighbors prior to knowing rectangle 
 * rect3_nfa:  orthogonality alignment check INSTEAD of parallel
 * make_markov3: Orthogonality to horiz/vert/depth lines

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
static grads ll_angle3( image3_double in,
                              struct coorlist3 ** list_p, void ** mem_p,
                              image3_double * modgrad, 
                  unsigned int n_bins,double alpha)
{
  unsigned int m,n,p,x,y,z,adr,i;
  double com1,com2,gx,gy,norm,norm2;
 
  int list_count = 0;
  struct coorlist3 * list;
  struct coorlist3 ** range_l_s; /* array of pointers to start of bin list */
  struct coorlist3 ** range_l_e; /* array of pointers to end of bin list */
  struct coorlist3 * start;
  struct coorlist3 * end;
  double max_grad = 0.0;

  /* check parameters */
  if( in == NULL || in->data == NULL || in->xsize == 0 || in->ysize == 0 || in->zsize ==0 )
    error("ll_angle3: invalid image.");
 
  if( list_p == NULL ) error("ll_angle3: NULL pointer 'list_p'.");
  if( mem_p == NULL ) error("ll_angle3: NULL pointer 'mem_p'.");
  if( modgrad == NULL ) error("ll_angle3: NULL pointer 'modgrad'.");
  if( n_bins == 0 ) error("ll_angle3: 'n_bins' must be positive.");

  /* image size shortcuts */
  m = in->zsize;
  n = in->ysize;
  p = in->xsize;
  //printf("\t Creating new grads...\n");
  fflush(stdout);

  grads angles = new_grads(p,n,m);
  /* get memory for the image of gradient modulus */
  *modgrad = new_image3_double(p,n,m);

  //printf("\t Creating coorlists...\n");
  //fflush(stdout);
  /* get memory for "ordered" list of pixels */
  list = (struct coorlist3 *) calloc( (size_t) (m*n*p), sizeof(struct coorlist3) );
  *mem_p = (void *) list;
  range_l_s = (struct coorlist3 **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
  range_l_e = (struct coorlist3 **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
  if( list == NULL || range_l_s == NULL || range_l_e == NULL )
    error("ll_angle3: not enough memory in list or range.");
  for(i=0;i<n_bins;i++) range_l_s[i] = range_l_e[i] = NULL;


  //printf("\t Nullifying boundaries...\n");
  //fflush(stdout);

  /* 'undefined' on the down and right boundaries 
   * undefined on the z-axis/depth boundaries, aka 'outer'
   */
  //Let y=(n-1) for 'down boundary'.  Cycle all x and z
  for(x=0;x<p;x++) for(z=0;z<m;z++){angles->az->data[ (z) + m*( (x) + p*(n-1) ) ] = NOTDEF;
                    angles->el->data[ (z) + m*( (x) + p*(n-1) ) ] = NOTDEF;}
  //Let x=(p-1), for 'right boundary'.  Cycle all y and z
  for(y=0;y<n;y++) for(z=0;z<m;z++){angles->az->data[ (z) + m*( (p-1) + p*(y) ) ] = NOTDEF;
                    angles->el->data[ (z) + m*( (p-1) + p*(y) ) ] = NOTDEF;}
  //Let z=(m-1), for 'out boundary'.  Cycle all x and z
  for(x=0;x<p;x++) for(y=0;y<n;y++){angles->az->data[ (m-1) + m*( (x) + p*(y) ) ] = NOTDEF;
                    angles->el->data[ (m-1) + m*( (x) + p*(y) ) ] = NOTDEF;}
 

int j,k,ha,hb;
double ax,ay,az,an;
  int X;  /* x image size */
  int Y;  /* y image size */
  int Z;
Z=m;
Y=n;
X=p;
double beta=alpha;
int nb=16;
//G1 matrices
int bitdepth=16;
double imgtemp;
int imgSize=X*Y*Z;
int longueur=ceil(log(10)*beta);
/*longueur=wid;*/
int largeur=longueur;

double Mg, Md, coeff;
double Mgx,Mdx,Mgy,Mdy,Mgz,Mdz;
int h,xx,yy,zz,hx,hy,hz;
//Loop2: compute m terms
printf("Get gradx,y,z\n");fflush(stdout);

double *img1= (double *) malloc( X * Y * Z *  sizeof(double) );

double *gradx1= (double *) malloc( X * Y * Z *  sizeof(double) );
double *gradx2= (double *) malloc( X * Y * Z *  sizeof(double) );
double *grady1= (double *) malloc( X * Y * Z *  sizeof(double) );
double *grady2= (double *) malloc( X * Y * Z *  sizeof(double) );
double *gradz1= (double *) malloc( X * Y * Z *  sizeof(double) );
double *gradz2= (double *) malloc( X * Y * Z *  sizeof(double) );

for(i=0;i<imgSize;i++)
{
    //memcost reduction, will remove 
    img1[i]=(double)in->data[i];
    if(img1[i]<1.) img1[i]=1.;
    //temp fill to improve access
    gradx1[i]=(double)in->data[i];
    gradx2[i]=(double)in->data[i];
    grady1[i]=(double)in->data[i];
    grady2[i]=(double)in->data[i];
    gradz1[i]=(double)in->data[i];
    gradz2[i]=(double)in->data[i];

}

//printf("DATA SUCCESSFULLY STORED \n");fflush(stdout);

//printf("Loop 1.... \n");fflush(stdout);
double startT,endT;
startT=omp_get_wtime();
#pragma omp parallel default(none) shared(X,Y,Z,largeur,beta,img1,gradx1,grady1,gradz1) private(i,j,k,h,xx,yy,zz,coeff,Mdx,Mdy,Mdz)   
{
#pragma omp for 
for(k=0;k<Z;k++){
    //if((k%10)==0) {printf("(%d/%d)...",k,Z);fflush(stdout);}
    for(j=0;j<Y;j++) {
        for(i=0;i<X;i++) {
        //Notationally: Let gradx1=Hxy, gradx2 = (Hxy)z
        //Notationally: Let grady1=Hyz, grady2 = (Hyz)x
        //Notationally: Let gradz1=Hzx, gradz2 = (Hzx)y
        //sum over the latter letter in order to get azimuth from x,y
        //For z, difference is verticle to get elevation 
        /*GRADX*/       
        Mdx=0.;  Mdy=0.;  Mdz=0.; 
        for(h=-largeur;h<=largeur;h++)
        {     
            xx = (i+h)>0?(i+h):0;
            xx = xx<(X-1)?xx:(X-1);
            yy = (j+h)>0?(j+h):0;
            yy = yy<(Y-1)?yy:(Y-1);
            zz = (k+h)>0?(k+h):0;
            zz = zz<(Z-1)?zz:(Z-1);
            coeff=exp(-(double) abs(h)/beta);
            Mdx+=coeff*img1[k +Z*(i +X*yy)];
            Mdy+=coeff*img1[zz+Z*(i +X*j )];
            Mdz+=coeff*img1[k +Z*(xx+X*j )];
        }
        gradx1[k+Z*(i+X*j)]=Mdx;
        grady1[k+Z*(i+X*j)]=Mdy;
        gradz1[k+Z*(i+X*j)]=Mdz;
        }
    }
}
}
endT=omp_get_wtime();
//printf("Parallel: %f seconds\n",endT-startT);fflush(stdout);

//printf("Loop 2.... \n");fflush(stdout);
startT = omp_get_wtime();

#pragma omp parallel default(none) shared(X,Y,Z,largeur,beta,gradx1,grady1,gradz1,gradx2,grady2,gradz2) private(i,j,k,h,xx,yy,zz,coeff,Mdx,Mdy,Mdz)   
{
#pragma omp for 
for(k=0;k<Z;k++){
    //if((k%10)==0) {printf("(%d/%d)...",k,Z);fflush(stdout);}
    for(j=0;j<Y;j++) {
        for(i=0;i<X;i++) {
        //Notationally: Let gradx1=Hxy, gradx2 = (Hxy)z
        //Notationally: Let grady1=Hyz, grady2 = (Hyz)x
        //Notationally: Let gradz1=Hzx, gradz2 = (Hzx)y
        //sum over the latter letter in order to get azimuth from x,y
        //For z, difference is verticle to get elevation 
        /*GRADX*/       
        Mdx=0.;  Mdy=0.;  Mdz=0.; 
        for(h=-largeur;h<=largeur;h++)
        {     
            xx = (i+h)>0?(i+h):0;
            xx = xx<(X-1)?xx:(X-1);
            yy = (j+h)>0?(j+h):0;
            yy = yy<(Y-1)?yy:(Y-1);
            zz = (k+h)>0?(k+h):0;
            zz = zz<(Z-1)?zz:(Z-1);
            coeff=exp(-(double) abs(h)/beta);
            Mdx+=coeff*gradx1[zz+Z*(i +X*j)];
            Mdy+=coeff*grady1[k +Z*(xx+X*j )];
            Mdz+=coeff*gradz1[k +Z*(i +X*yy)];
        }
        gradx2[k+Z*(i+X*j)]=Mdx;
        grady2[k+Z*(i+X*j)]=Mdy;
        gradz2[k+Z*(i+X*j)]=Mdz;
        }
    }
}
}
endT=omp_get_wtime();
//printf("Sequential: %f seconds\n",endT-startT);fflush(stdout);

//printf("Loop 3.... \n");fflush(stdout);

double an2;
#pragma omp parallel default(none) shared(X,Y,Z,largeur,beta,gradx2,grady2,gradz2,angles,modgrad) private(i,j,k,h,xx,yy,zz,coeff,Mdx,Mdy,Mdz,Mgx,Mgy,Mgz,ay,ax,az,adr,an,an2)   
{
#pragma omp for 
for(k=0;k<Z;k++){
    //if((k%10)==0) {printf("(%d/%d)...",k,Z);fflush(stdout);}
    for(j=0;j<Y;j++) {
        for(i=0;i<X;i++) {
        Mdx=0.; Mgx=0.;
        Mdy=0.; Mgy=0.;
        Mdz=0.; Mgz=0.;
        for(h=-largeur;h<=largeur;h++)
        {     
            xx = (i+h)>0?(i+h):0;
            xx = xx<(X-1)?xx:(X-1);
            yy = (j+h)>0?(j+h):0;
            yy = yy<(Y-1)?yy:(Y-1);
            zz = (k+h)>0?(k+h):0;
            zz = zz<(Z-1)?zz:(Z-1);
            coeff=exp(-(double) abs(h)/beta);
            if(h<0)
            {
                Mgx+=coeff*gradx2[k +Z*(xx+X*j)];
                Mgy+=coeff*grady2[k +Z*(i +X*yy )];
                Mgz+=coeff*gradz2[zz+Z*(i +X*j )];
            }
            if(h>0)
            {
                Mdx+=coeff*gradx2[k +Z*(xx+X*j)];
                Mdy+=coeff*grady2[k +Z*(i +X*yy )];
                Mdz+=coeff*gradz2[zz+Z*(i +X*j )];
            }
        }
        //SWAP AX AND AY per LSD ordering 
        ay=(double)log(Mdx/Mgx);
        ax=(double)log(Mdy/Mgy);
        az=(double)log(Mdz/Mgz);
        adr = (unsigned int)  k+Z*(j*X+i);
        an= (double)sqrt(ax*ax + ay*ay + az*az); 
        an2= (double)sqrt(ax*ax + ay*ay); 
        (*modgrad)->data[adr] =  an; /* store gradient norm */
    

        if( an <= 0.0 ) /* norm too small, gradient no defined */
        {
          angles->az->data[adr] = NOTDEF;
          angles->el->data[adr] = NOTDEF;
        }
        else
        {
          /* gradient angle computation */
	  //caution: if ax is near 0, will cause a crash  
	  if(an2<=0.0) angles->az->data[adr] =0.0;
	  else angles->az->data[adr] = atan2(ay,ax);

          angles->el->data[adr] = acos(az/an);
          /* look for the maximum of the gradient */
        } 
        //printf("\naz: %.2f, el: %.2f, an: %.2f\n",angles->az->data[adr],angles->el->data[adr],an);fflush(stdout);
    }
    }
    }
}
//MAXGRAD RACE CONDITION ERROR 
for(k=0;k<Z;k++){
    //if((k%10)==0) {printf("(%d/%d)...",k,Z);fflush(stdout);}
    for(j=0;j<Y;j++) {
        for(i=0;i<X;i++) {
            an= (double) (*modgrad)->data[k+Z*(j*X+i)]; 
	    if( an > max_grad ) max_grad = an;
}}}

free((void *) gradx1);
free((void *) grady1);
free((void *) gradz1);
free((void *) img1);
free((void *) gradx2);
free((void *) grady2);
free((void *) gradz2);
printf("Computing Histogram\n");fflush(stdout);
int i0;
  /* compute histogram of gradient values */
  for(x=0;x<X-1;x++)
  {
    for(y=0;y<Y-1;y++)
      {
        for(z=0;z<Z-1;z++)
    {
        norm = (*modgrad)->data[z+m*(x+y*p)];
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
        range_l_e[i0]->z = (int) z;
        range_l_e[i0]->next = NULL;
        }
      }
  }

  //printf("\t Ordering pixels...\n");
  //fflush(stdout);
  /* Make the list of pixels (almost) ordered by norm value.
     It starts by the larger bin, so the list starts by the
     pixels with the highest gradient value. Pixels would be ordered
     by norm value, up to a precision given by max_grad/n_bins.
   */
  
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
  free( (void *) range_l_s );
  free( (void *) range_l_e );
  int err=0;

  printf("LLGRAD COMPLETED\n");fflush(stdout);
  return angles;
}
static grads ll_angle3_backup( image3_double in,
                              struct coorlist3 ** list_p, void ** mem_p,
                              image3_double * modgrad, 
                  unsigned int n_bins,double alpha)
{
  unsigned int m,n,p,x,y,z,adr,i;
  double com1,com2,gx,gy,norm,norm2;
 
  int list_count = 0;
  struct coorlist3 * list;
  struct coorlist3 ** range_l_s; /* array of pointers to start of bin list */
  struct coorlist3 ** range_l_e; /* array of pointers to end of bin list */
  struct coorlist3 * start;
  struct coorlist3 * end;
  double max_grad = 0.0;

  /* check parameters */
  if( in == NULL || in->data == NULL || in->xsize == 0 || in->ysize == 0 || in->zsize ==0 )
    error("ll_angle3: invalid image.");
 
  if( list_p == NULL ) error("ll_angle3: NULL pointer 'list_p'.");
  if( mem_p == NULL ) error("ll_angle3: NULL pointer 'mem_p'.");
  if( modgrad == NULL ) error("ll_angle3: NULL pointer 'modgrad'.");
  if( n_bins == 0 ) error("ll_angle3: 'n_bins' must be positive.");

  /* image size shortcuts */
  m = in->zsize;
  n = in->ysize;
  p = in->xsize;
  //printf("\t Creating new grads...\n");
  fflush(stdout);

  grads angles = new_grads(p,n,m);
  /* get memory for the image of gradient modulus */
  *modgrad = new_image3_double(p,n,m);

  //printf("\t Creating coorlists...\n");
  //fflush(stdout);
  /* get memory for "ordered" list of pixels */
  list = (struct coorlist3 *) calloc( (size_t) (m*n*p), sizeof(struct coorlist3) );
  *mem_p = (void *) list;
  range_l_s = (struct coorlist3 **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
  range_l_e = (struct coorlist3 **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
  if( list == NULL || range_l_s == NULL || range_l_e == NULL )
    error("ll_angle3: not enough memory in list or range.");
  for(i=0;i<n_bins;i++) range_l_s[i] = range_l_e[i] = NULL;


  //printf("\t Nullifying boundaries...\n");
  //fflush(stdout);

  /* 'undefined' on the down and right boundaries 
   * undefined on the z-axis/depth boundaries, aka 'outer'
   */
  //Let y=(n-1) for 'down boundary'.  Cycle all x and z
  for(x=0;x<p;x++) for(z=0;z<m;z++){angles->az->data[ (z) + m*( (x) + p*(n-1) ) ] = NOTDEF;
                    angles->el->data[ (z) + m*( (x) + p*(n-1) ) ] = NOTDEF;}
  //Let x=(p-1), for 'right boundary'.  Cycle all y and z
  for(y=0;y<n;y++) for(z=0;z<m;z++){angles->az->data[ (z) + m*( (p-1) + p*(y) ) ] = NOTDEF;
                    angles->el->data[ (z) + m*( (p-1) + p*(y) ) ] = NOTDEF;}
  //Let z=(m-1), for 'out boundary'.  Cycle all x and z
  for(x=0;x<p;x++) for(y=0;y<n;y++){angles->az->data[ (m-1) + m*( (x) + p*(y) ) ] = NOTDEF;
                    angles->el->data[ (m-1) + m*( (x) + p*(y) ) ] = NOTDEF;}
 

int j,k,ha,hb;
double ax,ay,az,an;
  int X;  /* x image size */
  int Y;  /* y image size */
  int Z;
Z=m;
Y=n;
X=p;
double beta=alpha;
int nb=16;
//G1 matrices
int bitdepth=16;
double imgtemp;
int imgSize=X*Y*Z;
int longueur=ceil(log(10)*beta);
/*longueur=wid;*/
int largeur=longueur;

double Mg, Md, coeff;
float Mgx,Mdx,Mgy,Mdy,Mgz,Mdz;
int xx,yy,zz,hx,hy,hz;
//Loop2: compute m terms
printf("Get gradx,y,z\n");fflush(stdout);

float *img1;
img1= (float *) malloc( X * Y * Z *  sizeof(float) );
for(i=0;i<imgSize;i++)
{
    img1[i]=(float)in->data[i];
    if(img1[i]<1.) img1[i]=1.;
}

for(k=0;k<Z;k++)
{
    if((k%10)==0) {printf("(%d/%d)...",k,Z);fflush(stdout);}
    for(j=0;j<Y;j++)
    {
        for(i=0;i<X;i++)
        {
     
        /*GRADX*/       
        Mdx=0.; Mgx=0.;
        Mdy=0.; Mgy=0.;
        Mdz=0.; Mgz=0.;
        for(hx=-largeur;hx<=largeur;hx++)
        {
            xx = (i+hx)>0?(i+hx):0;
            xx = xx<(X-1)?xx:(X-1);
            for(hy=-largeur;hy<=largeur;hy++)
            {   
                yy = (j+hy)>0?(j+hy):0;
                yy = yy<(Y-1)?yy:(Y-1);
                for(hz=-largeur;hz<=largeur;hz++)
                {
                    zz = (k+hz)>0?(k+hz):0;
                    zz = zz<(Z-1)?zz:(Z-1);
                    coeff=exp(-((float) (abs(hx)+abs(hy)+abs(hz)))/beta);

                    //imgtemp=in->data[zz+Z*(xx+X*yy)];
                    //if(imgtemp<1.) imgtemp=1.;
                    imgtemp=img1[zz+Z*(xx+X*yy)]*coeff;

                    
                    if(hx<0) Mgx+=imgtemp;
                    if(hx>0) Mdx+=imgtemp;
                    
                    if(hy<0) Mgy+=imgtemp;
                    if(hy>0) Mdy+=imgtemp;
                    
                    if(hz<0) Mgz+=imgtemp;
                    if(hz>0) Mdz+=imgtemp;
                }
            }
        }
        
        ax=(double)log(Mdx/Mgx);
        ay=(double)log(Mdy/Mgy);
        az=(double)log(Mdz/Mgz);
        /*COMPUTE GRADS AND STORE DATA*/
        adr = (unsigned int)  k+Z*(j*X+i);
        
        an= sqrt(ax*ax + ay*ay + az*az); 
        //an=(double) sqrt((double)ax * (double)ax + 
        //   (double)ay * (double)ay + 
        //   (double)az * (double)az);
        (*modgrad)->data[adr] =  an; /* store gradient norm */
        if( an <= 0.0 ) /* norm too small, gradient no defined */
        {
        angles->az->data[adr] = NOTDEF;
        angles->el->data[adr] = NOTDEF;
        }
        else
        {
        /* gradient angle computation */
        angles->az->data[adr] = atan2( ax,-1*ay);
        angles->el->data[adr] = acos(az/an);
        /* look for the maximum of the gradient */
        if( an > max_grad ) max_grad = an;
        } 
        //printf("\naz: %.2f, el: %.2f, an: %.2f\n",angles->az->data[adr],angles->el->data[adr],an);fflush(stdout);
    }
    }
}
printf("Computing Histogram\n");fflush(stdout);
int i0;
  /* compute histogram of gradient values */
  for(x=0;x<X-1;x++)
  {
    for(y=0;y<Y-1;y++)
      {
        for(z=0;z<Z-1;z++)
    {
        norm = (*modgrad)->data[z+m*(x+y*p)];
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
        range_l_e[i0]->z = (int) z;
        range_l_e[i0]->next = NULL;
        }
      }
  }

  //printf("\t Ordering pixels...\n");
  //fflush(stdout);
  /* Make the list of pixels (almost) ordered by norm value.
     It starts by the larger bin, so the list starts by the
     pixels with the highest gradient value. Pixels would be ordered
     by norm value, up to a precision given by max_grad/n_bins.
   */
  
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
  free( (void *) range_l_s );
  free( (void *) range_l_e );
  int err=0;
  return angles;
}
/*
static grads ll_angle3_write( image3_double in,
                              struct coorlist3 ** list_p, void ** mem_p,
                              image3_double * modgrad, 
                  unsigned int n_bins,double alpha)
{
  unsigned int m,n,p,x,y,z,adr,i;
  double com1,com2,gx,gy,norm,norm2;
 
  int list_count = 0;
  struct coorlist3 * list;
  struct coorlist3 ** range_l_s; // array of pointers to start of bin list 
  struct coorlist3 ** range_l_e; // array of pointers to end of bin list 
  struct coorlist3 * start;
  struct coorlist3 * end;
  double max_grad = 0.0;

  // check parameters 
  if( in == NULL || in->data == NULL || in->xsize == 0 || in->ysize == 0 || in->zsize ==0 )
    error("ll_angle3: invalid image.");
 
  if( list_p == NULL ) error("ll_angle3: NULL pointer 'list_p'.");
  if( mem_p == NULL ) error("ll_angle3: NULL pointer 'mem_p'.");
  if( modgrad == NULL ) error("ll_angle3: NULL pointer 'modgrad'.");
  if( n_bins == 0 ) error("ll_angle3: 'n_bins' must be positive.");

  // image size shortcuts 
  m = in->zsize;
  n = in->ysize;
  p = in->xsize;
  //printf("\t Creating new grads...\n");
  fflush(stdout);

  grads angles = new_grads(p,n,m);
  // get memory for the image of gradient modulus 
  *modgrad = new_image3_double(p,n,m);

  //printf("\t Creating coorlists...\n");
  //fflush(stdout);
  // get memory for "ordered" list of pixels 
  list = (struct coorlist3 *) calloc( (size_t) (m*n*p), sizeof(struct coorlist3) );
  *mem_p = (void *) list;
  range_l_s = (struct coorlist3 **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
  range_l_e = (struct coorlist3 **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
  if( list == NULL || range_l_s == NULL || range_l_e == NULL )
    error("ll_angle3: not enough memory in list or range.");
  for(i=0;i<n_bins;i++) range_l_s[i] = range_l_e[i] = NULL;


  //printf("\t Nullifying boundaries...\n");
  //fflush(stdout);

  // 'undefined' on the down and right boundaries 
  // undefined on the z-axis/depth boundaries, aka 'outer'
   
  //Let y=(n-1) for 'down boundary'.  Cycle all x and z
  for(x=0;x<p;x++) for(z=0;z<m;z++){angles->az->data[ (z) + m*( (x) + p*(n-1) ) ] = NOTDEF;
                    angles->el->data[ (z) + m*( (x) + p*(n-1) ) ] = NOTDEF;}
  //Let x=(p-1), for 'right boundary'.  Cycle all y and z
  for(y=0;y<n;y++) for(z=0;z<m;z++){angles->az->data[ (z) + m*( (p-1) + p*(y) ) ] = NOTDEF;
                    angles->el->data[ (z) + m*( (p-1) + p*(y) ) ] = NOTDEF;}
  //Let z=(m-1), for 'out boundary'.  Cycle all x and z
  for(x=0;x<p;x++) for(y=0;y<n;y++){angles->az->data[ (m-1) + m*( (x) + p*(y) ) ] = NOTDEF;
                    angles->el->data[ (m-1) + m*( (x) + p*(y) ) ] = NOTDEF;}
 

int j,k,ha,hb;
double ax,ay,az,am,an,ap;
  int X;  // x image size 
  int Y;  // y image size 
  int Z;
Z=m;
Y=n;
X=p;
double beta=alpha;
int nb=16;
//G1 matrices
int bitdepth=16;
double imgtemp;


printf("Allocate memory \n");fflush(stdout);
int imgSize=X*Y*Z;
FILE * gradx, grady, gradz, gradx1, grady1, gradz1; 
gradx = fopen("gradx.dat","w");
ftruncate(fileno(gradx),imgSize*sizeof(double));
grady = fopen("grady.dat","w");
ftruncate(fileno(gradx),imgSize*sizeof(double));
gradz = fopen("gradz.dat","w");
ftruncate(fileno(gradx),imgSize*sizeof(double));

//Loop1: ensure image is normalized to [1,bitdepth], with clipping 
//       i.e., remove zero pixels to minimum bit detph of 1  
printf("Loop1: clip img\n");fflush(stdout);

int longueur=ceil(log(10)*beta);
int largeur=longueur;
//M matrices
double Mx, My, Mz, coeffa, coeffb, coeff;
int xha,yha,zha,xhb,yhb,zhb;
int adr,xadr, yadr,zadr;
//Loop2: compute m terms
printf("Loop2\n");fflush(stdout);
for(k=0;k<Z;k++)
{
    for(j=0;j<Y;j++)
    {
        for(i=0;i<X;i++)
        {
            //buffering for write is opposed by the non-contiguous 3d indexing          
            Mx=0.;  My=0.;  Mz=0.;
            for(ha=-largeur;ha<=largeur;ha++)
            {
                xha=min1(max1(i+ha,0),X-1);
                yha=min1(max1(j+ha,0),Y-1);
                coeffa=exp(-(double) abs(ha)/beta);
                for(hb=-largeur;hb<=largeur;hb++)
                {
                    zhb=min1(max1(k+hb,0),Z-1);
                    yhb=min1(max1(j+hb,0),Y-1);
                    coeffb=exp(-(double) abs(hb)/beta);
                    coeff = sqrt(coeffa*coeffa + coeffb*coeffb);
                    xadr =  (int) (zhb + Z*( i     + X*yha  ));
                    yadr =  (int) (zhb + Z*( xha   + X*j    ));
                    zadr =  (int) (k   + Z*( xha   + X*yha  ));
                    Mx+=max1(coeff,coeff*(in->data[xadr]));
                    My+=max1(coeff,coeff*(in->data[yadr]));
                    Mz+=max1(coeff,coeff*(in->data[zadr]));
                }
            }

            adr = (int) (k+Z*(i+X*j));
            fseek(gradx1,adr*sizeof(double),SEEK_SET);
            fwrite(&Mx,sizeof(double),1,gradx1);
            fseek(grady1,adr*sizeof(double),SEEK_SET);
            fwrite(&My,sizeof(double),1,grady1);
            fseek(gradz1,adr*sizeof(double),SEEK_SET);
            fwrite(&Mz,sizeof(double),1,gradz1);
        }
    }
}
fclose(gradx1);fclose(grady1);fclose(gradz1);
//Loop3: Compute G=log(m/m) terms
double Mxg, Myg, Mzg, Mxd, Myd, Mzd;
int xha1, xha2, yha1, yha2, yhb1, yhb2, zhb1, zhb2;
printf("Loop3\n");fflush(stdout);

for(k=0;k<Z;k++)
{
    for(j=0;j<Y;j++)
    {
        for(i=0;i<X;i++)
        {
            Mxg=0;  Mxd=0;  Myg=0;  Myd=0;  Mzg=0;  Mzd=0;
            for(ha=1;ha<=largeur;ha++)
            {
        
                //printf("\t ha\n");fflush(stdout);
                xha1 = max1(i-ha,0);
                            yha1 = max1(j-ha,0);
                            xha2 = min1(i+ha,X-1);
                        yha2 = min1(j+ha,Y-1);
                                coeffa=exp(-(double) abs(ha)/beta);
                                for(hb=1;hb<=largeur;hb++)
                                {
                    
                    //printf("\t hb\n");fflush(stdout);
                    yhb1 = max1(j-hb,0);
                    zhb1 = max1(k-hb,0);
                    yhb2 = min1(j+hb,Y-1);
                    zhb2 = min1(k+hb,Z-1);
                                        coeffb=exp(-(double) abs(hb)/beta);
                                        coeff = sqrt(coeffa*coeffa + coeffb*coeffb);  
                
                    //printf("\t Mx\n");fflush(stdout);
                    Mxg+=coeff*gradx1[zhb1 + Z*(i + X*yha1)   ];
                    Mxd+=coeff*gradx1[zhb2 + Z*(i + X*yha2)   ];

                    Myg+=coeff*grady1[zhb1 + Z*(xha1 + X*j)];
                    Myd+=coeff*grady1[zhb2 + Z*(xha2 + X*j)];
                    
                    Mzg+=coeff*gradz1[k    + Z*(xha1 + X*yhb1)];
                    Mzd+=coeff*gradz1[k    + Z*(xha1 + X*yhb2)];    
                }   
            }

            //printf("\t gradx: (%d,%d,%d) / (%d,%d,%d)\n",i,j,k,X,Y,Z);fflush(stdout);
            gradx[k+Z*(i+j*X)]=log(Mxd/Mxg);
            grady[k+Z*(i+j*X)]=log(Myd/Myg);
            gradz[k+Z*(i+j*X)]=log(Mzd/Mzg);
        }

    }
}
//Loop4: Compute GR norm and angle images

printf("Loop4\n");fflush(stdout);
double goodcount=0;
double totalcount=0;
for(k=0;k<Z;k++)
{
    for(i=0;i<X;i++)
    {
        for(j=0;j<Y;j++)
        {
            adr = (unsigned int)  k+Z*(j*X+i);
            az=gradz[adr];
            ay=gradx[adr];
            ax=grady[adr];
            an=(double) sqrt((double)ax * (double)ax + 
                     (double)ay * (double)ay + 
                     (double)az * (double)az);
            norm=an;
            (*modgrad)->data[adr] = norm; // store gradient norm 
            if (ax==0.00 && ay==0.00 && az==0.00) goodcount++;             
            if( norm <= 0.0 ) // norm too small, gradient no defined 
            {
                angles->az->data[adr] = NOTDEF;
                angles->el->data[adr] = NOTDEF;
            }
            else
            {
                // gradient angle computation 
                ap=atan2((double) ax,-(double) ay);
                angles->az->data[adr] = ap;
                am=acos((double) az/(double) an);
                angles->el->data[adr] = am;
                // look for the maximum of the gradient 
                if( norm > max_grad ) max_grad = norm;
            } 
        
        }
    }
}
int i0;
  // compute histogram of gradient values 
  for(x=0;x<X-1;x++)
  {
    for(y=0;y<Y-1;y++)
      {
        for(z=0;z<Z-1;z++)
    {
        norm = (*modgrad)->data[z+m*(x+y*p)];
        // store the point in the right bin according to its norm 
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
        range_l_e[i0]->z = (int) z;
        range_l_e[i0]->next = NULL;
        }
      }
  }

  //printf("\t Ordering pixels...\n");
  //fflush(stdout);
  // Make the list of pixels (almost) ordered by norm value.
    // It starts by the larger bin, so the list starts by the
     //pixels with the highest gradient value. Pixels would be ordered
     //by norm value, up to a precision given by max_grad/n_bins.
   
  
  // Make the list of pixels (almost) ordered by norm value.
    // It starts by the larger bin, so the list starts by the
    // pixels with the highest gradient value. Pixels would be ordered
    // by norm value, up to a precision given by max_grad/n_bins.
   
  
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
  free( (void *) range_l_s );
  free( (void *) range_l_e );
  int err=0;
  err = munmap(gradx,ARRAY_SIZE);
  if(err != 0) error("LL_angle3: unmap x");
  err = munmap(grady,ARRAY_SIZE);
  if(err != 0) error("LL_angle3: unmap y");
  err = munmap(gradz,ARRAY_SIZE);
  if(err != 0) error("LL_angle3: unmap z");
  err = munmap(gradx1,ARRAY_SIZE);
  if(err != 0) error("LL_angle3: unmap x1");
  err = munmap(grady1,ARRAY_SIZE);
  if(err != 0) error("LL_angle3: unmap y1");
  err = munmap(gradz1,ARRAY_SIZE);
  if(err != 0) error("LL_angle3: unmap z1");
  //
  //free ((void *) gradx);
  //free((void *) grady);
  //free((void *) gradz);
  //free((void *) imgfilt);
  //free((void *) gradx1);
 // free((void *) grady1);
  //free((void *) gradz1);
  
  //return angles;
}
*/
static grads ll_angle3_orig( image3_double in,
                              struct coorlist3 ** list_p, void ** mem_p,
                              image3_double * modgrad, 
                  unsigned int n_bins,double alpha)
{
  unsigned int m,n,p,x,y,z,adr,i;
  double com1,com2,gx,gy,norm,norm2;
 
  int list_count = 0;
  struct coorlist3 * list;
  struct coorlist3 ** range_l_s; /* array of pointers to start of bin list */
  struct coorlist3 ** range_l_e; /* array of pointers to end of bin list */
  struct coorlist3 * start;
  struct coorlist3 * end;
  double max_grad = 0.0;

  /* check parameters */
  if( in == NULL || in->data == NULL || in->xsize == 0 || in->ysize == 0 || in->zsize ==0 )
    error("ll_angle3: invalid image.");
 
  if( list_p == NULL ) error("ll_angle3: NULL pointer 'list_p'.");
  if( mem_p == NULL ) error("ll_angle3: NULL pointer 'mem_p'.");
  if( modgrad == NULL ) error("ll_angle3: NULL pointer 'modgrad'.");
  if( n_bins == 0 ) error("ll_angle3: 'n_bins' must be positive.");

  /* image size shortcuts */
  m = in->zsize;
  n = in->ysize;
  p = in->xsize;
  //printf("\t Creating new grads...\n");
  fflush(stdout);

  grads angles = new_grads(p,n,m);
  /* get memory for the image of gradient modulus */
  *modgrad = new_image3_double(p,n,m);

  //printf("\t Creating coorlists...\n");
  //fflush(stdout);
  /* get memory for "ordered" list of pixels */
  list = (struct coorlist3 *) calloc( (size_t) (m*n*p), sizeof(struct coorlist3) );
  *mem_p = (void *) list;
  range_l_s = (struct coorlist3 **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
  range_l_e = (struct coorlist3 **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
  if( list == NULL || range_l_s == NULL || range_l_e == NULL )
    error("ll_angle3: not enough memory in list or range.");
  for(i=0;i<n_bins;i++) range_l_s[i] = range_l_e[i] = NULL;


  //printf("\t Nullifying boundaries...\n");
  //fflush(stdout);

  /* 'undefined' on the down and right boundaries 
   * undefined on the z-axis/depth boundaries, aka 'outer'
   */
  //Let y=(n-1) for 'down boundary'.  Cycle all x and z
  for(x=0;x<p;x++) for(z=0;z<m;z++){angles->az->data[ (z) + m*( (x) + p*(n-1) ) ] = NOTDEF;
                    angles->el->data[ (z) + m*( (x) + p*(n-1) ) ] = NOTDEF;}
  //Let x=(p-1), for 'right boundary'.  Cycle all y and z
  for(y=0;y<n;y++) for(z=0;z<m;z++){angles->az->data[ (z) + m*( (p-1) + p*(y) ) ] = NOTDEF;
                    angles->el->data[ (z) + m*( (p-1) + p*(y) ) ] = NOTDEF;}
  //Let z=(m-1), for 'out boundary'.  Cycle all x and z
  for(x=0;x<p;x++) for(y=0;y<n;y++){angles->az->data[ (m-1) + m*( (x) + p*(y) ) ] = NOTDEF;
                    angles->el->data[ (m-1) + m*( (x) + p*(y) ) ] = NOTDEF;}
 

int j,k,ha,hb;
double ax,ay,az,am,an,ap;
  int X;  /* x image size */
  int Y;  /* y image size */
  int Z;
Z=m;
Y=n;
X=p;
double beta=alpha;
int nb=16;
//G1 matrices
int bitdepth=16;
double imgtemp;
int imgSize=X*Y*Z;
double *imgfilt;
imgfilt= (double *) calloc( (size_t) imgSize, sizeof(double) );

double * gradx; double * grady; double * gradz;
double * gradx1; double * grady1; double * gradz1;

gradx= (double *) calloc( (size_t) imgSize, sizeof(double) );
grady= (double *) calloc( (size_t) imgSize, sizeof(double) );
gradz= (double *) calloc( (size_t) imgSize, sizeof(double) );

gradx1= (double *) calloc( (size_t) imgSize, sizeof(double) );
grady1= (double *) calloc( (size_t) imgSize, sizeof(double) );
gradz1= (double *) calloc( (size_t) imgSize,sizeof(double) );


/*
int mmapset = -1;
int ARRAY_SIZE = imgSize*sizeof(double);

imgfilt= (double *) mmap(NULL, ARRAY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, mmapset, 0);
if(imgfilt ==  MAP_FAILED) error("LL_angles3: mapping imgfilt");
gradx= (double *) mmap(NULL, ARRAY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,mmapset , 0);
if(gradx ==  MAP_FAILED) error("LL_angles3: mapping x");
grady= (double *) mmap(NULL, ARRAY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,mmapset, 0);
if(grady ==  MAP_FAILED) error("LL_angles3: mapping y");
gradz= (double *) mmap(NULL, ARRAY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, mmapset, 0);
if(gradz ==  MAP_FAILED) error("LL_angles3: mapping z");
gradx1= (double *) mmap(NULL, ARRAY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,mmapset , 0);
if(gradx1 == MAP_FAILED) error("LL_angles3: mapping 1");
grady1= (double *) mmap(NULL, ARRAY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,mmapset, 0);
if(grady1 == MAP_FAILED) error("LL_angles3: mapping y1");
gradz1= (double *) mmap(NULL, ARRAY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, mmapset, 0);
if(gradz1 == MAP_FAILED) error("LL_angles3: mapping z1");
*/

if(gradx==NULL || grady==NULL || gradz==NULL)
    error("ERROR: could not allocate gradx memory ");
if(gradx1==NULL || grady1==NULL || gradz1==NULL)
    error("ERROR: could not allocate gradx1 memory ");




//G matrices

//G matrices   
/*
double * gradx;
int imgSize=X*Y*Z;
gradx= (double *) calloc( (size_t) imgSize, sizeof(double) );
double * grady;
grady= (double *) calloc( (size_t) imgSize, sizeof(double) );
double * gradz;
gradz= (double *) calloc( (size_t) imgSize, sizeof(double) );
int bitdepth=16;
double *imgfilt;
imgfilt= (double *) calloc( (size_t) imgSize, sizeof(double) );
double imgtemp;
*/
//Loop1: ensure image is normalized to [1,bitdepth], with clipping 
//       i.e., remove zero pixels to minimum bit detph of 1  
//printf("reload\n");fflush(stdout);
//if (imgfilt2 == NULL) {freopen("new.data","wb",imgfilt2);}
printf("Loop1: clip img\n");fflush(stdout);
for(i=0;i<imgSize;i++)
{
    //if(i % 100 == 0 ){printf("%d, ",i);fflush(stdout);}
    imgtemp=in->data[i];
    if(imgtemp<1.)  imgfilt[i]=1.;
    else    imgfilt[i]=imgtemp;
}

//M matrices 
int longueur=ceil(log(10)*beta);
/*longueur=wid;*/
int largeur=longueur;
/*
double * gradx1;
gradx1= (double *) calloc( (size_t) imgSize, sizeof(double) );
double * grady1;
grady1= (double *) calloc( (size_t) imgSize, sizeof(double) );
double * gradz1;
gradz1= (double *) calloc( (size_t) imgSize,sizeof(double) );
*/
//if(gradx1==NULL || grady1==NULL || gradz1==NULL)
//  error("ERROR: could not allocate gradx1 memory ");

double Mx, My, Mz, coeffa, coeffb, coeff;
int xha,yha,zha,xhb,yhb,zhb;
//Loop2: compute m terms
printf("Loop2\n");fflush(stdout);
for(k=0;k<Z;k++)
{
    for(j=0;j<Y;j++)
    {
        for(i=0;i<X;i++)
        {
            Mx=0.;  My=0.;  Mz=0.;
            for(ha=-largeur;ha<=largeur;ha++)
            {
                xha=min1(max1(i+ha,0),X-1);
                yha=min1(max1(j+ha,0),Y-1);
                coeffa=exp(-(double) abs(ha)/beta);
                for(hb=-largeur;hb<=largeur;hb++)
                {
                    zhb=min1(max1(k+hb,0),Z-1);
                    yhb=min1(max1(j+hb,0),Y-1);
                    coeffb=exp(-(double) abs(hb)/beta);
                    coeff = sqrt(coeffa*coeffa + coeffb*coeffb);
                    Mx+=coeff*imgfilt[ zhb + Z*( i   + X*yha  )];
                    My+=coeff*imgfilt[ zhb + Z*( xha + X*j )];
                    Mz+=coeff*imgfilt[ k   + Z*( xha + X*yhb  )];
                }
            }
            gradx1[k+Z*(i+X*j)]=Mx;
            grady1[k+Z*(i+X*j)]=My;
            gradz1[k+Z*(i+X*j)]=Mz;
        }
    }
}
//Loop3: Compute G=log(m/m) terms
double Mxg, Myg, Mzg, Mxd, Myd, Mzd;
int xha1, xha2, yha1, yha2, yhb1, yhb2, zhb1, zhb2;

printf("Loop3\n");fflush(stdout);

for(k=0;k<Z;k++)
{
    for(j=0;j<Y;j++)
    {
        for(i=0;i<X;i++)
        {
            Mxg=0;  Mxd=0;  Myg=0;  Myd=0;  Mzg=0;  Mzd=0;
            for(ha=1;ha<=largeur;ha++)
            {
        
                //printf("\t ha\n");fflush(stdout);
                xha1 = max1(i-ha,0);
                            yha1 = max1(j-ha,0);
                            xha2 = min1(i+ha,X-1);
                        yha2 = min1(j+ha,Y-1);
                                coeffa=exp(-(double) abs(ha)/beta);
                                for(hb=1;hb<=largeur;hb++)
                                {
                    
                    //printf("\t hb\n");fflush(stdout);
                    yhb1 = max1(j-hb,0);
                    zhb1 = max1(k-hb,0);
                    yhb2 = min1(j+hb,Y-1);
                    zhb2 = min1(k+hb,Z-1);
                                        coeffb=exp(-(double) abs(hb)/beta);
                                        coeff = sqrt(coeffa*coeffa + coeffb*coeffb);  
                
                    //printf("\t Mx\n");fflush(stdout);
                    Mxg+=coeff*gradx1[zhb1 + Z*(i + X*yha1)   ];
                    Mxd+=coeff*gradx1[zhb2 + Z*(i + X*yha2)   ];

                    Myg+=coeff*grady1[zhb1 + Z*(xha1 + X*j)];
                    Myd+=coeff*grady1[zhb2 + Z*(xha2 + X*j)];
                    
                    Mzg+=coeff*gradz1[k    + Z*(xha1 + X*yhb1)];
                    Mzd+=coeff*gradz1[k    + Z*(xha1 + X*yhb2)];    
                }   
            }

            //printf("\t gradx: (%d,%d,%d) / (%d,%d,%d)\n",i,j,k,X,Y,Z);fflush(stdout);
            gradx[k+Z*(i+j*X)]=log(Mxd/Mxg);
            grady[k+Z*(i+j*X)]=log(Myd/Myg);
            gradz[k+Z*(i+j*X)]=log(Mzd/Mzg);
        }

    }
}
//Loop4: Compute GR norm and angle images

printf("Loop4\n");fflush(stdout);
double goodcount=0;
double totalcount=0;
for(k=0;k<Z;k++)
{
    for(i=0;i<X;i++)
    {
        for(j=0;j<Y;j++)
        {
            adr = (unsigned int)  k+Z*(j*X+i);
            az=gradz[adr];
            ay=gradx[adr];
            ax=grady[adr];
            an=(double) sqrt((double)ax * (double)ax + 
                     (double)ay * (double)ay + 
                     (double)az * (double)az);
            norm=an;
            (*modgrad)->data[adr] = norm; /* store gradient norm */
            if (ax==0.00 && ay==0.00 && az==0.00) goodcount++;             
            if( norm <= 0.0 ) /* norm too small, gradient no defined */
            {
                angles->az->data[adr] = NOTDEF;
                angles->el->data[adr] = NOTDEF;
            }
            else
            {
                /* gradient angle computation */
                ap=atan2((double) ax,-(double) ay);
                angles->az->data[adr] = ap;
                am=acos((double) az/(double) an);
                angles->el->data[adr] = am;
                /* look for the maximum of the gradient */
                if( norm > max_grad ) max_grad = norm;
            } 
        
        }
    }
}
int i0;
  /* compute histogram of gradient values */
  for(x=0;x<X-1;x++)
  {
    for(y=0;y<Y-1;y++)
      {
        for(z=0;z<Z-1;z++)
    {
        norm = (*modgrad)->data[z+m*(x+y*p)];
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
        range_l_e[i0]->z = (int) z;
        range_l_e[i0]->next = NULL;
        }
      }
  }

  //printf("\t Ordering pixels...\n");
  //fflush(stdout);
  /* Make the list of pixels (almost) ordered by norm value.
     It starts by the larger bin, so the list starts by the
     pixels with the highest gradient value. Pixels would be ordered
     by norm value, up to a precision given by max_grad/n_bins.
   */
  
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
  free( (void *) range_l_s );
  free( (void *) range_l_e );
  int err=0;
  /*
  err = munmap(gradx,ARRAY_SIZE);
  if(err != 0) error("LL_angle3: unmap x");
  err = munmap(grady,ARRAY_SIZE);
  if(err != 0) error("LL_angle3: unmap y");
  err = munmap(gradz,ARRAY_SIZE);
  if(err != 0) error("LL_angle3: unmap z");
  err = munmap(gradx1,ARRAY_SIZE);
  if(err != 0) error("LL_angle3: unmap x1");
  err = munmap(grady1,ARRAY_SIZE);
  if(err != 0) error("LL_angle3: unmap y1");
  err = munmap(gradz1,ARRAY_SIZE);
  if(err != 0) error("LL_angle3: unmap z1");
  */
  
  free ((void *) gradx);
  free((void *) grady);
  free((void *) gradz);
  free((void *) imgfilt);
  free((void *) gradx1);
  free((void *) grady1);
  free((void *) gradz1);
  
  return angles;
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


/** Is point (x,y,z) aligned to angle theta, up to precision 'prec', in quat space?
 * NOTE: ALIGNMENT IS ORTHOGONAL 
 */
//static int isaligned3( int x, int y, int z, grads angles, angles3 theta, double prec )
static int isaligned3(double grads_az,double grads_el,double theta_az,double theta_el,double prec)
{
  
  //printf("\t\t\t\t Initializing IsAligned3...\n");
  //fflush(stdout);
  /* check parameters */
  /*
  if( angles->az == NULL || angles->az->data == NULL ||
      angles->el == NULL || angles->el->data == NULL)
    error("isaligned3: invalid images in grads 'angles'.");
  if( x < 0 || y < 0 || z < 0 || 
      x>=(int)angles->az->xsize || y>=(int)angles->az->ysize || z>=(int)angles->az->zsize)
    error("isaligned3: (x,y,z) out of the image.");
  */
  if( prec < 0.0 ) error("isaligned3: 'prec' must be positive.");

  //printf("\t\t\t\t Getting angles...\n");
  //fflush(stdout);
  /* angles at pixel (x,y,z) */
  //double az = angles->az->data[ z + angles->az->zsize*(x + y * angles->az->xsize) ]; 
  //double el = angles->el->data[ z + angles->el->zsize*(x + y * angles->el->xsize) ];
  
  /* pixels whose level-line angle is not defined
     are considered as NON-aligned */

  //printf("\t\t\t\t Check defined...\n");
  //fflush(stdout);
  

  //if( theta_az == NULL || theta_el == NULL ) error("isaligned3: invalid theta angles");
  if( grads_az == NOTDEF ) return FALSE;
  if( grads_el == NOTDEF ) return FALSE;  /* there is no need to call the function
                                      'double_equal' here because there is
                                      no risk of problems related to the
                                      comparison doubles, we are only
                                      interested in the exact NOTDEF value */
  
  //Store angles into structure and compute quaternions 
  
  //printf("\t\t\t\t Store indexed angles, making a %.2f %.2f...\n",az,el);
  //fflush(stdout);
  //angles3 a;

  //printf("\t\t\t\t Store indexed angles, storing %.2f %.2f...\n",az,el);
  //fflush(stdout);
  //a=new_angles3(az,el);
  
  //printf("\t\t\t\t Compute quaternion diff.  Az %.2f, El %.2f...\n",az,el);
  //fflush(stdout);
  //theta is well-defined, a is ill-defined
  //double diff = quat_diff(a,theta);
  
  ///double  dl1[3] =  {cos(a->az)*sin(a->el), sin(a->az)*sin(a->el),cos(a->el)};
  ///double  dl2[3] =  {cos(az)*sin(el), sin(az)*sin(el),cos(el)};
  ///for(int i=0;i<3;i++) dot+=dl1[i]*dl2[i];
  ///double diff =  fabs(dot);

  //double diff = (gsl_sf_cos(grads_az)*gsl_sf_sin(grads_el)*gsl_sf_cos(theta_az)*gsl_sf_sin(theta_el)) + (   gsl_sf_sin(grads_az)*gsl_sf_sin(grads_el)*gsl_sf_sin(theta_az)*gsl_sf_sin(theta_el)) + (  gsl_sf_cos(grads_el)*gsl_sf_cos(theta_el) )   ;
  //double diff = (cos(grads_az)*sin(grads_el)*cos(theta_az)*sin(theta_el)) + (   sin(grads_az)*sin(grads_el)*sin(theta_az)*sin(theta_el)) + (  cos(grads_el)*cos(theta_el) )   ;
  double sGel,sTel,cGel,cTel;
  sincos(grads_el,&sGel,&cGel);
  sincos(theta_el,&sTel,&cTel);
  double diff = sGel*sTel*cos(grads_az-theta_az) + cGel*cTel;
  
  double a = grads_az;
  double theta = theta_az;
  theta -= a;
  if( theta < 0.0 ) theta = -theta;
  if( theta > M_3_2_PI )
    {
      theta -= M_2__PI;
      if( theta < 0.0 ) theta = -theta;
    }

  if  (theta <= prec)
  	return (1.-fabs(diff))<=sin(prec);
  else
	return 0;
  //double diff = sin(grads_el)*sin(theta_el)*cos(grads_az-theta_az) + cos(grads_el)*cos(theta_el);

  //NOTE: (a*b)=|a||b|cos(theta), so require |cos(theta)|<cos(pi/2 - tol)

//printf("\t\t\t\t Free and return comparison ...\n");
  //fflush(stdout);
  //free_angles3(a); 
  
  /*the quaternion difference is cosine-distributed on [-1,1], 
   * and so we seek |diff|<cos(pi/2 - prec) as orthogonal alignment near 0*/
  //Note that there are twice as many octants in 3D space as quadrants in 2D,
  //and so precision must be half of the 2D precision on input 
  //printf("\t\t\t\t DIFF: %.8f, LIMIT: %.2f\n",diff,cos(M_PI/2. - prec));
  //fflush(stdout);
  //return acos(1-diff)<=prec;
  //return diff <= cos(M_PI/2.- prec);
  //printf("%.2f\n",prec);fflush(stdout);
  //return fabs(diff) <= cos(M_PI/2. - prec);
  //
}

static int isorthogonal3(double grads_az,double grads_el,double theta_az,double theta_el,double prec)
{
  if( prec < 0.0 ) error("isaligned3: 'prec' must be positive.");
  if( grads_az == NOTDEF ) return FALSE;
  if( grads_el == NOTDEF ) return FALSE;
  double diff = sin(grads_el)*sin(theta_el)*cos(grads_az-theta_az) + cos(grads_el)*cos(theta_el);
  return fabs(diff)<=sin(prec);
  //
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

/** Rectangular prim (non-rolling) structure: line segment with square width.
 */
struct rect3
{
  double x1,y1,z1,x2,y2,z2;  /* first and second point of the line segment */
  double length,width1,width2;        /* rectangle width */
  double x,y,z;          /* center of the rectangle */
  angles3 theta;        /* az/el angle as struct angle3 */
  double dl[3],dw1[3],dw2[3];        /* dr,daz,del are each 3lenth vectors oriented parallel/orthogonal to the line segment */
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

static void rect3_copy(struct rect3 * in, struct rect3 * out)
{
  /* check parameters */
  if( in == NULL || out == NULL ) error("rect3_copy: invalid 'in' or 'out'.");

  /* copy values */
  out->x1 = in->x1;
  out->y1 = in->y1;
  out->z1 = in->z1;
  out->x2 = in->x2;
  out->y2 = in->y2;
  out->z2 = in->z2; 
  out->length = in->length;
  out->width1 = in->width1;
  out->width2 = in->width2;
  out->x = in->x;
  out->y = in->y;
  out->z = in->z;
  out->theta = in->theta;
  for(int i=0;i<3;i++)
  {
    out->dl[i]  = in->dl[i];
    out->dw1[i] = in->dw1[i];
    out->dw2[i] = in->dw2[i];
  }
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

    CLSD UPDATE: 
    For the prism, we must iterate over Z, fixing X and iterating all Y, 
    a standard extension of the 2D case for each z index
    Hence we explore column-by-column as 
    a sub-loop of exploring row-by-row.
    Therefore, in addition to ye/ys pairs, we require a ze/zs pair 
    \verbatim


                    vx[5],vy[5]vz[5]
                          *   *
                         *       *
                        *           *
                   *               ye
                  *                :  *
             vx[4],vy[4]vz[4]      :     *
                        *              :        *
                       *          x,y,z        *
                          *        :              *
                             *     :            vx[6],vy[6]vz[6]
                                *  :                *
                                   ys              *
                                  (ze)*           *
                                  :     *       *
                                 :        *   *
              vx[1],vy[1]vz[1]     :         vx[7],vy[7],vz[7]
                 *   *           :
                *       *       :
               *           *  (zs)-undefined
              *               ye
             *                :  *
        vx[0],vy[0]vz[0]      :     *
               *              :        *
            xs    *          x,y,z        *
                :    *        :              *
                   :    *     :            vx[2],vy[2]vz[2]
                      :   *   :                *
        y               :     ys              *
        ^                  :     *           *
        |                    :      *       *
        |                      xe      *   *
        +---> x                   vx[3],vy[3]vz[3]
       -
      -
     z

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

    CLSD UPDATE:
    Similarly, the first 'row' to be explored must have the smaller z value.
    The corners must be specified with additional vz[], and dimensionality increased
    from 4 vertices to 8 vertices.  'Clockwise' order is therefore invalid, 
    Since we introduce 'xs' < 'xe', let us pick the lowest z point.
    Since there is no rotation, z points come in paired level sets.
    Therefore, we can now pick the smallest x value for this z, label it xs, 
    and set vertex zero as (xs,ys,zsmallest).  Then order v1,v2,v3,v4 follows
    the 2D  process on the XY-plane for column-by-column exploration. 
 */
typedef struct
{
  double vx[4];  /* rectangle's corner X coordinates in circular order */
  double vy[4];  /* rectangle's corner Y coordinates in circular order */
  int x,y;       /* coordinates of currently explored pixel */
  double xd,yd; // double coordinate of rotated explored pixel
  int xt,yt; // coorinate of rotated explored pixel
  int xspan, yspan; // rotationally-invarient explorable dimensions
  //new
  int update;
  double dl[2]; //vector for rotating the x coordinate
  double dn[2]; //vector for rotating the y coordinate
  double ys,ye;
} rect_iter;

typedef struct
{
  double vx[8];  /* rectangle's corner X coordinates in circular order */
  double vy[8];  /* rectangle's corner Y coordinates in circular order */
  double vz[8]; 
  int x,y,z; // pixel coordinates in original image frame
  double xd,yd,zd; // double coordinate of rotated explored pixel
  int xt,yt,zt; // coorinate of rotated explored pixel
  int xspan, yspan,zspan; // rotationally-invarient explorable dimensions
  //new
  int update;
  double dl[3];
  double dw1[3];
  double dw2[3];
} rect3_iter;

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

static void ri3_del(rect3_iter * iter)
{
  if( iter == NULL ) error("ri_del: NULL iterator.");
  free( (void *) iter );
}
/*----------------------------------------------------------------------------*/
/** Check if the iterator finished the full iteration.

    See details in \ref rect_iter
 */
static int ri_end_old(rect_iter * i)
{
  if( i == NULL ) error("ri_end: NULL iterator.");
  return (double)(i->x) > i->vx[2];
}
static int ri_end(rect_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri_end: NULL iterator.");

  /* if the current x value is larger than the largest
     x value in the rectangle (vx[2]), we know the full
     exploration of the rectangle is finished. */
  return (i->xt > i->xspan) && (i->yt > i->yspan) ;
}

static int ri3x_end(rect3_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri3x_end: NULL iterator.");

  /* if the current x value is larger than the largest
     x value in the plane for a fixed z (xe),
     exploration of the rectangle is finished. */

  return (i->xt > i->xspan) && (i->yt > i->yspan);
  //return (double)(i->x) > i->xe;
}
static int ri3z_end(rect3_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri3z_end: NULL iterator.");

  /* if the current z value is larger than the largest
     z value in the rectangle (vx[7]), we know the full
     exploration of the rectangle is finished.
     Note that vz[6]=vz[7] just as vx[2]=vx[3] in 2D */
  return (i->zt > i->zspan) && (i->xt > i->xspan) && (i->yt > i->yspan);
}
/*----------------------------------------------------------------------------*/
/** Increment a rectangle iterator.
    See details in \ref rect_itedn[0])) + ((double)(i->yt)*(i->dn[1]));
        if (i->y != (int) floor(i->yd)) {
                i->y = (int) floor(i->yd);
                i->update = 1;}
}*/


static void up_all(rect_iter * i)
{
        //x growth derivatives
        i->xd = i->vx[0] + ((double)(i->xt)*(i->dl[0])) + ((double)(i->yt)*(i->dl[1]));
        if (i->x != (int) ceil(i->xd)-1) {
                i->x = (int) ceil(i->xd)-1;
                i->update = 1;}
        //y growth derivatives
        i->yd = i->vy[0] + ((double)(i->xt)*(i->dn[0])) + ((double)(i->yt)*(i->dn[1]));
        if (i->y != (int) ceil(i->yd)) {
                i->y = (int) ceil(i->yd);
                i->update = 1;}
}

// Iterate pixels in the rotated plane, then project to the cartesian image
static void ri_inc(rect_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri_inc: NULL iterator.");
  i->update=0;
  //Increment integer-wise in the rotated plane 
  while(i->update==0)
  {
      // if not at end of exploration, inc. y to next 'col'
      if(!(ri_end(i))) {i->yt++;}
      //if at end of col, inc. x to next row
      while( (i->yt > i->yspan) && !(ri_end(i)) )
      {
        i->xt++;    
        if(ri_end(i)) {up_all(i); return;}
        //set yt to zero.  'ys' is f(v[0],xt)
        i->yt = 0; 
      }
      //Update cartesian pixel coordinate 
      up_all(i);
  }
}
static void ri_inc_old(rect_iter * i)
{
  if( i == NULL ) error("ri_inc: NULL iterator.");

  if( !ri_end(i) ) i->y++;
  while( (double) (i->y) > i->ye && !ri_end(i) )
    {
      i->x++;
      if( ri_end(i) ) return;

      if( (double) i->x < i->vx[3] )
        i->ys = inter_low((double)i->x,i->vx[0],i->vy[0],i->vx[3],i->vy[3]);
      else
        i->ys = inter_low((double)i->x,i->vx[3],i->vy[3],i->vx[2],i->vy[2]);
      if( (double)i->x < i->vx[1] )
        i->ye = inter_hi((double)i->x,i->vx[0],i->vy[0],i->vx[1],i->vy[1]);
      else
        i->ye = inter_hi((double)i->x,i->vx[1],i->vy[1],i->vx[2],i->vy[2]);
      i->y = (int) ceil(i->ys);
    }
}

/*----------------------------------------------------------------------------*/
/** Increment a rectangle iterator.
    See details in \ref rect_iter
 */
// Project the rotated plane pixel to cartesian pixels
static void up_all3(rect3_iter * i)
{
        //x growth derivatives
        i->xd = i->vx[0] + ((double)(i->xt)*(i->dl[0])) 
        + ((double)(i->yt)*(i->dl[1])) + ((double)(i->zt)*(i->dl[2]));
        if (i->x != (int) ceil(i->xd)-1) {
                i->x = (int) ceil(i->xd)-1;
                i->update = 1;}
        //y growth derivatives
        i->yd = i->vy[0] + ((double)(i->xt)*(i->dw1[0])) 
        + ((double)(i->yt)*(i->dw1[1])) + ((double)(i->zt)*(i->dw1[2]));
        if (i->y != (int) ceil(i->yd)) {
                i->y = (int) ceil(i->yd);
                i->update = 1;}
        //z growth derivatives 
        i->zd = i->vz[0] + ((double)(i->xt)*(i->dw2[0])) 
        + ((double)(i->yt)*(i->dw2[1])) + ((double)(i->zt)*(i->dw2[2]));
        if (i->z != (int) ceil(i->zd)) {
                i->z = (int) ceil(i->zd);
                i->update = 1;}

}

static void ri3_inc(rect3_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri_inc: NULL iterator.");
  i->update=0;
  //Increment integer-wise in the rotated plane 
  while(i->update==0)
  {
      //up_all3(i);
      // if not at end of exploration, inc. y to next 'col'
      if(!(ri3z_end(i))) i->zt++;
      
      //up_all3(i);
      //if at end of col, inc. x to next row
      while( (i->zt > i->zspan) && !(ri3z_end(i)) )
      {
        if(!(ri3z_end(i))) i->yt++;
        
        //up_all3(i);
        while( (i->yt > i->yspan) && !(ri3z_end(i)) )   
        {
            i->xt++;
            
            //up_all3(i);
            if(ri3z_end(i)){up_all3(i); return;}
            i->yt=0;
            //up_all3(i);
        }
        if(ri3z_end(i)) {up_all3(i); return;}
        //set yt to zero.  'ys' is f(v[0],xt)
        i->zt = 0;
        //up_all3(i);
      }
      //Update cartesian pixel coordinate 
      up_all3(i);
  }
}
// Iterate pixels in the rotated plane, then project to the cartesian image
static void ri3_inc_backup(rect3_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri_inc: NULL iterator.");
  i->update=0;
  //Increment integer-wise in the rotated plane 
  while(i->update==0)
  {
      //up_all3(i);
      // if not at end of exploration, inc. y to next 'col'
      if(!(ri3z_end(i))) i->zt++;
      
      //up_all3(i);
      //if at end of col, inc. x to next row
      while( (i->zt > i->zspan) && !(ri3z_end(i)) )
      {
        if(!(ri3z_end(i))) i->yt++;
        
        //up_all3(i);
        while( (i->yt > i->yspan) && !(ri3z_end(i)) )   
        {
            i->xt++;
            
            //up_all3(i);
            if(ri3z_end(i)){up_all3(i); return;}
            i->yt=0;
            //up_all3(i);
        }
        if(ri3z_end(i)) {up_all3(i); return;}
        //set yt to zero.  'ys' is f(v[0],xt)
        i->zt = 0;
        //up_all3(i);
      }
      //Update cartesian pixel coordinate 
      up_all3(i);
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
 
  i->x  = (int) ceil(i->vx[0]) - 1;
  i->y = (int) ceil(i->vy[0]);
  i->xspan =  dist(r->x1,r->y1,r->x2,r->y2); 
  i->yspan = r->width;
  i->xt = i->yt =  0;
  i->xd = i->yd = 0.;
  i->dl[0] = fabs(r->dx);
  i->dl[1] = fabs(r->dy);
  i->dn[0] = -1.*fabs(r->dy);
  i->dn[1] = fabs(r->dx);
  i->ys=i->ye = -DBL_MAX; 
  if(i->vy[0] < i->vy[2]){i->dn[0]*=-1.; i->dn[1]*=-1.;} 
  /* advance to the first pixel */
  ri_inc(i);

  return i;
}

/** Create and initialize a rectangle iterator.

    See details in \ref rect_iter
 */
static rect3_iter * ri3_ini(struct rect3 * r)
{
  double vx[8],vy[8],vz[8];
  int n,offset;
  rect3_iter * i;

  /* check parameters */
  if( r == NULL ) error("ri3_ini: invalid rectangle.");

  /* get memory */
  i = (rect3_iter *) malloc(sizeof(rect3_iter));
  if( i == NULL ) error("ri3_ini: Not enough memory.");

  /* build list of rectangle corners ordered
     in a circular way around the rectangle 
     with two z level sets (v0-v3 on lower z)
     NOTE: width1 is along the azimithal direction (xy plane), 
           width2 is along the elevation direction (xz plane),
       calculated in the polar coordinate fashion.
       yz-plane behaviors are 
    
  r = r +- dw1*(width1/2) +- dw2*(width2/2);
  r(0:2) use +dw1,  r(2:4) use -dw1
  r(0:4) use -dw2,  r(4:8) use +dw2
  */

  //lower z
  //printf("%.2f\n",r->dw1[0]); fflush(stdout);
  //printf("(%.1f, %.1f, %.1f, %.1f) \t",r->dw1[1],r->dw2[1],r->width1,r->width2);fflush(stdout);
  vx[0] = r->x1 + (r->dw1[0] * r->width1 / 2.0) - (r->dw2[0] * r->width2 / 2.0);
  vy[0] = r->y1 + (r->dw1[1] * r->width1 / 2.0) - (r->dw2[1] * r->width2 / 2.0);
  vz[0] = r->z1 + (r->dw1[2] * r->width1 / 2.0) - (r->dw2[2] * r->width2 / 2.0);

  vx[1] = r->x2 + (r->dw1[0] * r->width1 / 2.0) - (r->dw2[0] * r->width2 / 2.0);
  vy[1] = r->y2 + (r->dw1[1] * r->width1 / 2.0) - (r->dw2[1] * r->width2 / 2.0);
  vz[1] = r->z2 + (r->dw1[2] * r->width1 / 2.0) - (r->dw2[2] * r->width2 / 2.0);

  vx[2] = r->x2 - (r->dw1[0] * r->width1 / 2.0) - (r->dw2[0] * r->width2 / 2.0);
  vy[2] = r->y2 - (r->dw1[1] * r->width1 / 2.0) - (r->dw2[1] * r->width2 / 2.0);
  vz[2] = r->z2 - (r->dw1[2] * r->width1 / 2.0) - (r->dw2[2] * r->width2 / 2.0);

  vx[3] = r->x1 - (r->dw1[0] * r->width1 / 2.0) - (r->dw2[0] * r->width2 / 2.0);
  vy[3] = r->y1 - (r->dw1[1] * r->width1 / 2.0) - (r->dw2[1] * r->width2 / 2.0);
  vz[3] = r->z1 - (r->dw1[2] * r->width1 / 2.0) - (r->dw2[2] * r->width2 / 2.0);

  //upper z
  vx[4] = r->x1 + (r->dw1[0] * r->width1 / 2.0) + (r->dw2[0] * r->width2 / 2.0);
  vy[4] = r->y1 + (r->dw1[1] * r->width1 / 2.0) + (r->dw2[1] * r->width2 / 2.0);
  vz[4] = r->z1 + (r->dw1[2] * r->width1 / 2.0) + (r->dw2[2] * r->width2 / 2.0);

  vx[5] = r->x2 + (r->dw1[0] * r->width1 / 2.0) + (r->dw2[0] * r->width2 / 2.0);
  vy[5] = r->y2 + (r->dw1[1] * r->width1 / 2.0) + (r->dw2[1] * r->width2 / 2.0);
  vz[5] = r->z2 + (r->dw1[2] * r->width1 / 2.0) + (r->dw2[2] * r->width2 / 2.0);

  vx[6] = r->x2 - (r->dw1[0] * r->width1 / 2.0) + (r->dw2[0] * r->width2 / 2.0);
  vy[6] = r->y2 - (r->dw1[1] * r->width1 / 2.0) + (r->dw2[1] * r->width2 / 2.0);
  vz[6] = r->z2 - (r->dw1[2] * r->width1 / 2.0) + (r->dw2[2] * r->width2 / 2.0);

  vx[7] = r->x1 - (r->dw1[0] * r->width1 / 2.0) + (r->dw2[0] * r->width2 / 2.0);
  vy[7] = r->y1 - (r->dw1[1] * r->width1 / 2.0) + (r->dw2[1] * r->width2 / 2.0);
  vz[7] = r->z1 - (r->dw1[2] * r->width1 / 2.0) + (r->dw2[2] * r->width2 / 2.0);
  
  /* compute rotation of index of corners needed so that the first
     point has the smaller x from among the smaller z.
     if one side is vertical, thus two corners have the same smaller z
     value, the one with the largest x value is selected as the first 
     if one side is vertical, thus two corners have the same smaller x
     value, the one with the largest y value is selected as the first.
   */
  //there exist 4 unique orders for z (z1,z2,+dz,-dz)
  //and         4 unique orders for x (x1,x2,+dx,-dx)


  if( r->x1 < r->x2 && r->y1 <= r->y2 ) offset = 0;
  else if( r->x1 >= r->x2 && r->y1 < r->y2 ) offset = 1;
  else if( r->x1 > r->x2 && r->y1 >= r->y2 ) offset = 2;
  else offset = 3;
  int offsetz = 0;
  if (r->z1 > r->z2) offsetz=1; 

  /* apply rotation of index. */
  /*unsigned int m;
  for(n=0; n<4; n++)
    {
    for(m=0;m<2;m++)
    {
            i->vx[n+m*4] = vx[(offset+n)%4 + 2*((offsetz+m)%2)];
            i->vy[n+m*4] = vy[(offset+n)%4 + 2*((offsetz+m)%2)];
            i->vz[n+m*4] = vz[(offset+n)%4 + 2*((offsetz+m)%2)];
    }
    }
    */
  offset=1;
  double tx[8],ty[8], tz[8];
  //printf("%.0f...",vy[0]);fflush(stdout);

  while(vx[0]>vx[1] || vx[0]>vx[2] || vx[0]>vx[3])
  {
	  while(vx[0]>vx[1] || vx[0]>vx[2] || vx[0]>vx[3])
	  {
	    for(n=0;n<8;n++)
	    {
		tx[n]=vx[n];
		ty[n]=vy[n];
		tz[n]=vz[n];
	    }
	    for(n=0;n<4;n++)
	    {
		//z1
		vx[n] = tx[(offset+n)%4];
		vy[n] = ty[(offset+n)%4];
		vz[n] = tz[(offset+n)%4];
		//z2
		vx[n+4] = tx[(offset+n)%4+4];
		vy[n+4] = ty[(offset+n)%4+4];
		vz[n+4] = tz[(offset+n)%4+4];
	    }
	  }
	  if (vx[0]>vx[4])
	  {
	    for(n=0;n<8;n++)
	    {
		tx[n]=vx[n];
		ty[n]=vy[n];
		tz[n]=vz[n];
	    }
	    for(n=0;n<4;n++)
	    {

		//z1
		vx[n] = tx[n+4];
		vy[n] = ty[n+4];
		vz[n] = tz[n+4];
		//z2
		vx[n+4] = tx[n];
		vy[n+4] = ty[n];
		vz[n+4] = tz[n];
	    }
	  }
	  for(n=0;n<8;n++)
	  {
	      i->vx[n]=vx[n];
	      i->vy[n]=vy[n];
	      i->vz[n]=vz[n];
	  }
  }
  


  //printf("%.2f\t",vy[0]);fflush(stdout);


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
  i->x = (int) ceil(i->vx[0])-1;
  i->y = (int) ceil(i->vy[0]);
  //printf("(%d, %d, %d) \t",i->x,i->y,i->z);fflush(stdout);
  //printf("(%d, %.2f,) \t",i->y,i->vy[0]);fflush(stdout);
  i->z = (int) ceil(i->vz[0]);
  i->xspan = r->length; 
  i->yspan = r->width1;
  i->zspan = r->width2;
  i->xt = 0; i->yt = 0; i->zt = 0; i->xd = 0; i->yd = 0; i->zd = 0;
  

  double saz,caz,sel,cel;
  sincos(r->theta->az,&saz,&caz);
  sincos(r->theta->el,&sel,&cel);
  saz=fabs(saz);caz=fabs(caz);sel=fabs(sel);cel=fabs(cel);
  /*
  double caz,saz,cel,sel;
  caz = fabs(cos(r->theta->az));
  saz = fabs(sin(r->theta->az));
  cel = fabs(cos(r->theta->el));
  sel = fabs(sin(r->theta->el));
  */
  double  dl[3] =  {caz*sel, saz*sel,  cel};
  //double dw1[3] =  {-saz*sel, caz*sel, 0};
  double dw1[3] =  {-saz, caz, 0};
  double dw2[3] =  {caz*cel, saz*cel, -sel};
  
  for(n=0;n<3;n++) 
  {
    i->dl[n]  = dl[n];//fabs(r->dl[n]);
    i->dw1[n] = dw1[n];//fabs(r->dw1[n]);
    i->dw2[n] = dw2[n];//fabs(r->dw2[n]);
    //if(n==0) i->dw1[n] *= -1.;
    //if(n==2) i->dw2[n] *= -1.;
    //if(n==0 || n==1) i->dw2[n] *= -1.;
  }  
  if(i->vy[0] < i->vy[2]) for(n=0;n<3;n++) i->dw1[n]*=-1.; 
  if(i->vz[0] < i->vz[5]) for(n=0;n<3;n++) i->dw2[n]*=-1.;
  /* advance to the first pixel */
  ri3_inc(i);

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
  //printf("x: (%d, %d), vs: (%.2f, %.2f)\n",i->x,i->y,i->vx[2],i->vy[2]);fflush(stdout);
  ri_del(i); /* delete iterator */
  if(pts<minreg)
      return -1;
  else
  return nfa(pts,alg,rec->p,logNT,image,N); /* compute NFA value */
}

static double rect3_nfa(struct rect3 * rec, grads angles, double logNT,double *image,int N,int minreg)
{
  rect3_iter * i;
  int pts = 0;
  int alg = 0;

  /* check parameters */
  if( rec == NULL ) error("rect3_nfa: invalid rectangle.");
  if( angles == NULL ) error("rect3_nfa: invalid grads structure 'angles'.");
  if( angles->az->data ==NULL || angles->el->data ==NULL) error("rect3_nfa:invalid az/el images withing grads structure 'angles'");

  /* compute the total number of pixels and of aligned points in 'rec' */
  //printf("\n rec dz %.2f\n",rec->z2-rec->z1);
  int tester=0;
  double grads_az, grads_el;
  int xsize=(int) angles->az->xsize;
  int ysize=(int) angles->az->ysize;
  int zsize=(int) angles->az->zsize;
  double theta_az = rec->theta->az;
  double theta_el = rec->theta->el;
  double prec = rec->prec;
  //double* az = (double*) angles->az->data;
  //double* el = (double*) angles->el->data;
  
  /* 
  int* x;int* xtmp;
  int* y;int* ytmp;
  int* z;int* ztmp;
  printf("beginning pt count:\n");fflush(stdout);
  for(i=ri3_ini(rec); !ri3z_end(i); ri3_inc(i))
  {
  	if( i->x >= 0 && i->y >= 0 && i->z >=0 &&
        	i->x < xsize && i->y < ysize && i->z < zsize)
      	{
 		++pts;
		printf("(%d,%d,%d)...",i->x,i->y,i->z);fflush(stdout);
		if(pts==1) {
			xtmp = (int*) malloc(sizeof(int));
			ytmp = (int*) malloc(sizeof(int));
			ztmp = (int*) malloc(sizeof(int));
		}
		else{
			xtmp = realloc(x,pts*sizeof(int));
			ytmp = realloc(y,pts*sizeof(int));
			ztmp = realloc(z,pts*sizeof(int));
		}
		
		x = xtmp; x[pts] = (int) i->x;
		y = ytmp; y[pts] = (int) i->y;
		z = ztmp; z[pts] = (int) i->z;
	}
  }
  printf("\n");fflush(stdout);
  ri3_del(i); 

  if(pts<minreg)
      return -1;
  int j;

#pragma omp parallel default(none) shared(x,y,z,xsize,zsize,az,el,theta_az,theta_el,prec,pts,alg) private(j,grads_az,grads_el)   
 {
 #pragma omp for reduction(+:alg)
  for(j=0; j<=pts; j++) // rectangle iterator 
  {
        grads_az = az[ z[j] + zsize*(x[j] + y[j] * xsize) ]; 
        grads_el = el[ z[j] + zsize*(x[j] + y[j] * xsize) ];
        if(isaligned3(grads_az,grads_el, theta_az, theta_el, prec)) ++alg;
        //if( isaligned3(i->x, i->y,i->z, angles, rec->theta, rec->prec) ) ++alg; 
        // aligned points counter 
  }
  }
*/

  for(i=ri3_ini(rec); !ri3z_end(i); ri3_inc(i)) // rectangle iterator 
  {
  if( i->x >= 0 && i->y >= 0 && i->z >=0 &&
        i->x < xsize && i->y < ysize && i->z < zsize)
      {
        ++pts; // total number of pixels counter 
        grads_az = angles->az->data[ i->z + zsize*(i->x + i->y * xsize) ]; 
        grads_el = angles->el->data[ i->z + zsize*(i->x + i->y * xsize) ];
        if(isorthogonal3(grads_az,grads_el, theta_az, theta_el, prec)) ++alg;
        //if( isaligned3(i->x, i->y,i->z, angles, rec->theta, rec->prec) ) ++alg; 
        // aligned points counter 
      }
  }
  

  //if (tester==1) {printf("%d",pts);fflush(stdout);}
  ri3_del(i); /* delete iterator */
  //printf("\t\tpts %d, minreg %d, alg %d\n",pts,minreg,alg);  fflush(stdout);
  if(pts<minreg)
      return -1;
  else
  return nfa(pts,alg,rec->p,logNT,image,N); /* compute NFA value */
}

static double rect3_nfa_backup(struct rect3 * rec, grads angles, double logNT,double *image,int N,int minreg)
{
  rect3_iter * i;
  int pts = 0;
  int alg = 0;

  /* check parameters */
  if( rec == NULL ) error("rect3_nfa: invalid rectangle.");
  if( angles == NULL ) error("rect3_nfa: invalid grads structure 'angles'.");
  if( angles->az->data ==NULL || angles->el->data ==NULL) error("rect3_nfa:invalid az/el images withing grads structure 'angles'");

  /* compute the total number of pixels and of aligned points in 'rec' */
  //printf("\n rec dz %.2f\n",rec->z2-rec->z1);
  int tester=0;
  double grads_az, grads_el;
  int xsize=(int) angles->az->xsize;
  int ysize=(int) angles->az->ysize;
  int zsize=(int) angles->az->zsize;
  double theta_az = rec->theta->az;
  double theta_el = rec->theta->el;
  double prec = rec->prec;
  for(i=ri3_ini(rec); !ri3z_end(i); ri3_inc(i)) /* rectangle iterator */
  {
  //printf("(%d,%d,%d)",i->x,i->y,i->z);  fflush(stdout);  
  if( i->x >= 0 && i->y >= 0 && i->z >=0 &&
        i->x < xsize && i->y < ysize && i->z < zsize)
      {
    tester=1;
        ++pts; /* total number of pixels counter */
        
        grads_az = angles->az->data[ i->z + zsize*(i->x + i->y * xsize) ]; 
        grads_el = angles->el->data[ i->z + zsize*(i->x + i->y * xsize) ];
        if(isaligned3(grads_az,grads_el, theta_az, theta_el, prec)) ++alg;
        //if( isaligned3(i->x, i->y,i->z, angles, rec->theta, rec->prec) ) ++alg; 
        /* aligned points counter */
      }
  //if(tester==0){
      
  //if(i->y <0){printf("ylow: %d\t",i->y);fflush(stdout);}
      //if(i->x <0){printf("xlow\t");fflush(stdout);}
      //if(i->y <0){printf("ylow\t");fflush(stdout);}
      //if(i->z <0){printf("zlow\t");fflush(stdout);}
      //if(i->x >=angles->az->xsize){printf("xhigh\t");fflush(stdout);}
      //if(i->y >=angles->az->ysize){printf("yhigh\t");fflush(stdout);}
      //if(i->z >=angles->az->zsize){printf("zhigh\t\n");fflush(stdout);}}
  }
  //if (tester==1) {printf("%d",pts);fflush(stdout);}
  ri3_del(i); /* delete iterator */
  //printf("\t\tpts %d, minreg %d, alg %d\n",pts,minreg,alg);  fflush(stdout);
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


/** Compute region's angle as the principal inertia axis of the region.
    The following is the region inertia matrix A:
    @f[

        A = \left(\begin{array}{cc}
                                    Ixx & Ixy & Ixz\\
                                    Ixy & Iyy & Iyz\\
                    Ixz & Iyz & Izz
             \end{array}\right)

    @f]
    where
      Ixx =   sum_i G(i).((y_i - cy)^2 + (z_i - cz)^2)
      Iyy =   sum_i G(i).((x_i - cx)^2 + (z_i - cz)^2)
      Izz =   sum_i G(i).((x_i - cx)^2 + (y_i - cy)^2)
      Ixy = - sum_i G(i).(x_i - cx).(y_i - cy)
      Ixz = - sum_i G(i).(x_i - cx).(z_i - cz)
      Ixy = - sum_i G(i).(y_i - cy).(z_i - cz)
    and
    - G(i) is the gradient norm at pixel i, used as pixel's weight.
    - x_i and y_i are the coordinates of pixel i.
    - cx and cy are the coordinates of the center of th region.

    Eigendecomposition by GSL for real symmetric matrices.
    Returns az/el of the *principle* eigenvector for region, for 
    (orthogonal) alignment checks of new pixels.
    The orthogonality check means there is no orientation ambiguity.  
 */

static angles3 get_theta3( struct point3 * reg, int reg_size, double x, double y, double z,
                         image3_double modgrad, angles3 reg_angle, double prec )
{
  
  //printf("\t\t\t Initialize get_theta3...\n");
  fflush(stdout);
  double lambda,weight;
  double Ixx = 0.0;
  double Iyy = 0.0;
  double Izz = 0.0;
  double Ixy = 0.0;
  double Ixz = 0.0;
  double Iyz = 0.0;
  int i;

  /* check parameters */
  if( reg == NULL ) error("get_theta3: invalid region.");
  if( reg_size <= 1 ) error("get_theta3: region size <= 1.");
  if( modgrad == NULL || modgrad->data == NULL )
    error("get_theta3: invalid 'modgrad'.");
  if( prec < 0.0 ) error("get_theta3: 'prec' must be positive.");

  //printf("\t\t\t Inertia matrix, %d points...\n",reg_size);
  //fflush(stdout);
  /* compute inertia matrix */
  for(i=0; i<reg_size; i++)
    {
      weight = modgrad->data[reg[i].z + (reg[i].x + reg[i].y * modgrad->xsize) * modgrad->zsize ];
      Ixx += ( ( (double) reg[i].y - y ) * ( (double) reg[i].y - y ) + 
           ( (double) reg[i].z - z ) * ( (double) reg[i].z - z) )
          * weight;

      Iyy += ( ( (double) reg[i].x - x ) * ( (double) reg[i].x - x ) + 
           ( (double) reg[i].z - z ) * ( (double) reg[i].z - z) )
          * weight;
      Izz += ( ( (double) reg[i].y - y ) * ( (double) reg[i].y - y ) + 
           ( (double) reg[i].x - x ) * ( (double) reg[i].x - x) )
          * weight;  
      Ixy -= ( (double) reg[i].x - x ) * ( (double) reg[i].y - y ) * weight;
      Ixz -= ( (double) reg[i].x - x ) * ( (double) reg[i].z - z ) * weight;
      Iyz -= ( (double) reg[i].z - z ) * ( (double) reg[i].y - y ) * weight;
  
    }
  if( double_equal(Ixx,0.0) && double_equal(Iyy,0.0) && double_equal(Ixy,0.0) &&
    double_equal(Izz,0.0) && double_equal(Ixz,0.0) && double_equal(Iyz,0.0))
    error("get_theta3: null inertia matrix.");

  /*Gsl eigendecomposition on real symmetric data, for robustness 
   *'symmetric bidiagonalization and QR reduction method' per documentation 
   *eigenvalues accurate up to e_mach*||I||_2 
   */

  //printf("\t\t\t GSL computation...\n");
  //fflush(stdout);
  size_t dim = 3;
  gsl_eigen_symmv_workspace * ework = gsl_eigen_symmv_alloc(dim);
  gsl_matrix * Imat = gsl_matrix_alloc(dim,dim);
  gsl_matrix_set(Imat, 0, 0, Ixx);
  gsl_matrix_set(Imat, 1, 1, Iyy);
  gsl_matrix_set(Imat, 2, 2, Izz);
  gsl_matrix_set(Imat, 0, 1, Ixy);
  gsl_matrix_set(Imat, 1, 0, Ixy);
  gsl_matrix_set(Imat, 0, 2, Ixz);
  gsl_matrix_set(Imat, 2, 0, Ixz);
  gsl_matrix_set(Imat, 1, 2, Iyz);
  gsl_matrix_set(Imat, 2, 1, Iyz);
  gsl_vector * eval = gsl_vector_alloc(dim);
  gsl_matrix * evec = gsl_matrix_alloc(dim,dim);

  //printf("\t\t\t get principle evec...\n");
  //fflush(stdout);
  //Get principle eigenvector
  int out = gsl_eigen_symmv(Imat, eval, evec, ework);
  //for(i=0;i<3;i++) gsl_vector_set(eval,i,abs(gsl_vector_get(eval,i)));
  if(~gsl_vector_ispos(eval)) {for(i=0;i<3;i++) gsl_vector_set(eval,i,fabs(gsl_vector_get(eval,i)));}
  size_t idx = gsl_vector_min_index(eval);
  double xv = gsl_matrix_get(evec, 0, idx);
  double yv = gsl_matrix_get(evec, 1, idx);
  double zv = gsl_matrix_get(evec, 2, idx);
  //Get angularity of eigenvector.  Orientation is irrelevant for orthogonality.
  //double vnorm=sqrt(xv*xv + yv*yv + zv*zv);
  //printf("\t\t\t Get evec angle (%.2f, %.2f, %.2f)=%.2f...\n",xv,yv,zv,vnorm);
  //fflush(stdout);
  angles3 theta = line_angle3(0,0,0,xv,yv,zv);  
   
  //printf("\t\t\t Free GSL...\n");
  //fflush(stdout);
  //free memory 
  gsl_vector_free(eval);
  gsl_matrix_free(evec);
  gsl_matrix_free(Imat);
  gsl_eigen_symmv_free(ework);

  //printf("\t\t\t End GSL\n");
  //fflush(stdout);
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

static void region2rect3( struct point3 * reg, int reg_size,
                         image3_double modgrad, angles3 reg_angle,
                         double prec, double p, struct rect3 * rec )
{
  
  //printf("\t\t Initialize...\n");
  fflush(stdout);
  double x,y,z,dx,dy,dz;
  double l,w1,w2,weight,sum,l_min,l_max,w1_min,w1_max,w2_min,w2_max;
  angles3 theta;
  int i;

  /* check parameters */
  if( reg == NULL ) error("region2rect3: invalid region.");
  if( reg_size <= 1 ) error("region2rect3: region size <= 1.");
  if( modgrad == NULL || modgrad->data == NULL )
    error("region2rect3: invalid image 'modgrad'.");
  if( rec == NULL ) error("region2rect3: invalid 'rec'.");

  /* center of the region:

     It is computed as the weighted sum of the coordinates
     of all the pixels in the region. The norm of the gradient
     is used as the weight of a pixel. The sum is as follows:
       cx = \sum_i G(i).x_i
       cy = \sum_i G(i).y_i
     where G(i) is the norm of the gradient of pixel i
     and x_i,y_i are its coordinates.
   */

  //printf("\t\t Centering...\n");
  //fflush(stdout);
  x = 0; y = 0; z = 0; sum = 0.0;
  for(i=0; i<reg_size; i++)
    {
      weight = modgrad->data[ reg[i].z + (reg[i].x + reg[i].y * modgrad->xsize) 
          * modgrad->zsize ];
      x += (double) reg[i].x * weight;
      y += (double) reg[i].y * weight;
      z += (double) reg[i].z * weight;
      sum += weight;
    }
  if( sum <= 0.0 ) error("region2rect3: weights sum equal to zero.");
  x /= sum;
  y /= sum;
  z /= sum;

  //printf("\t\t Get theta3...\n");
  //fflush(stdout);
  /* theta */
  theta = get_theta3(reg,reg_size,x,y,z,modgrad,reg_angle,prec);

  /* length and width:

     (dx,dy,dx) are computed by standard sphereical-cartesian relations.
     'l' is defined along the primary axis, and 'w1' , 'w2' along the tangents.
    Accordingly, where c and s abbreviate sine,cosine:
    lhat  = dr/dr  = [caz*sel, saz*sel,  cel]
    w1hat = dr/daz = [   -saz,     cel,    0]
    w2hat = dr/del = [caz*cel, saz*cel, -sel] 
     And projections are carried out as l=<r,lhat>

     The length of the rectangle goes from l_min to l_max, where l_min
     and l_max are the minimum and maximum values of l in the region.
     Analogously, the width is selected from w_min to w_max, where
     w_min and w_max are the minimum and maximum of w for the pixels
     in the region.
   */

  //printf("\t\t Get Derivatives...\n");
  //fflush(stdout);
  //
  double saz,caz,sel,cel;
  sincos(theta->az,&saz,&caz);
  sincos(theta->el,&sel,&cel);
  /* 
  double caz,saz,cel,sel;
  caz = cos(theta->az);
  saz = sin(theta->az);
  cel = cos(theta->el);
  sel = sin(theta->el);
  */
  double  dl[3] =  {caz*sel, saz*sel,  cel};
  //double dw1[3] =  {-saz*sel, caz*sel, 0};
  double dw1[3] =  {-saz, caz, 0};
  double dw2[3] =  {caz*cel, saz*cel, -sel};

  //double dw1[3] =  {   -saz,     cel,    0};
  //double dw2[3] =  {caz*cel, saz*cel, -sel};

  //printf("\t\t Get length...\n");
  //fflush(stdout);
  l_min = 0.;l_max = 0.;w1_min = 0.;w1_max =0.; w2_min =0.; w2_max = 0.0;
  for(i=0; i<reg_size; i++)
    {
      l =  ( (double) reg[i].x - x) * dl[0] + ( (double) reg[i].y - y) * dl[1] + 
         ( (double) reg[i].z - z) * dl[2];
      w1=  ( (double) reg[i].x - x) * dw1[0] + ( (double) reg[i].y - y) * dw1[1] + 
         ( (double) reg[i].z - z) * dw1[2];
      w2=  ( (double) reg[i].x - x) * dw2[0] + ( (double) reg[i].y - y) * dw2[1] + 
         ( (double) reg[i].z - z) * dw2[2];

      if( l > l_max ) l_max = l;
      if( l < l_min ) l_min = l;
      if( w1 > w1_max ) w1_max = w1;
      if( w1 < w1_min ) w1_min = w1;
      if( w2 > w2_max ) w2_max = w2;
      if( w2 < w2_min ) w2_min = w2;
    }

  //printf("\t\t Store...\n");
  //fflush(stdout);
  /* store values */
  rec->x1 = x + l_min * dl[0];
  rec->y1 = y + l_min * dl[1];
  rec->z1 = z + l_min * dl[2];
  rec->x2 = x + l_max * dl[0];
  rec->y2 = y + l_max * dl[1];
  rec->z2 = z + l_max * dl[2];
  rec->length = l_max - l_min;
  rec->width1 = w1_max - w1_min;
  rec->width2 = w2_max - w2_min;
  rec->x = x;
  rec->y = y;
  rec->z = z;
  rec->theta = theta;
  
  for(i=0;i<3;i++)
  {
    rec->dl[i]  = (double)dl[i];
    rec->dw1[i] = (double)dw1[i];
    rec->dw2[i] = (double)dw2[i];
  }
  
  //rec->dl  = dl;
  //rec->dw1 = dw1;
  //rec->dw2 = dw2;
  rec->prec = prec;
  rec->p = p;

  /* we impose a minimal width of one pixel

     A sharp horizontal or vertical step would produce a perfectly
     horizontal or vertical region. The width computed would be
     zero. But that corresponds to a one pixels width transition in
     the image.
   */
  if( rec->width1 < 1.0 ) rec->width1 = 1.0;
  if( rec->width2 < 1.0 ) rec->width2 = 1.0;
  //printf("(%.0f, %.0f, %.0f) \t",rec->x1,rec->y1,rec->z1);fflush(stdout);
  //printf("\t\t End Calc\n");
  //fflush(stdout);
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


/** Build a region of pixels that share the same angle, up to a
    tolerance 'prec', starting at point (x,y).
 */
//grads angles
static void region3_grow(int x, int y,int z, grads angles, 
			 struct point3 * reg,
                         int * reg_size, angles3 * reg_angle, 
			 image3_char used,double prec ,int NOUT)
{
  
  //printf("\t\t Initialize...\n");
  fflush(stdout);
    
  double sumdx,sumdy,sumdz;
  int xx,yy,zz,i;

  /* check parameters */
  if( x < 0 || y < 0 || z<0 || x >= (int) used->xsize 
          || y >= (int) used->ysize || z >= (int) used->zsize)
    error("region3_grow: (x,y,z) out of the image.");
  if( angles->az == NULL || angles->az->data == NULL )
    error("region3_grow: invalid grads 'angles' or image 'az'.");
  if( reg == NULL ) error("region3_grow: invalid 'reg'.");
  if( reg_size == NULL ) error("region3_grow: invalid pointer 'reg_size'.");
  if( reg_angle == NULL ) error("region3_grow: invalid pointer 'reg_angle'.");
  if( used == NULL || used->data == NULL )
    error("region3_grow: invalid image 'used'.");

  //printf("\t\t Get first point...\n");
  //fflush(stdout);
  /* first point of the region */
  *reg_size = 1;
  reg[0].x = x;
  reg[0].y = y;
  reg[0].z = z;
  /*regions angles*/
 
  //printf("\t\t Get angles...\n");
  //fflush(stdout);
  int xsize = (int)used->xsize;
  int ysize = (int)used->ysize;
  int zsize = (int)used->zsize;
  //(*reg_angle)->az = angles->az->data[z + (x+y*xsize)*zsize]; 
  //(*reg_angle)->el = angles->el->data[z + (x+y*xsize)*zsize]; 
  
  double reg_az, reg_el;
  reg_az = angles->az->data[z + (x+y*xsize)*zsize]; 
  reg_el = angles->el->data[z + (x+y*xsize)*zsize]; 
  //calc_quat(&(*reg_angle));
  //printf("\t\t Get derivatives...\n");
  //fflush(stdout);

  double saz,caz,sel,cel;	    
  sincos(reg_az,&saz,&caz);
  sincos(reg_el,&sel,&cel);
  sumdx = caz*sel;
  sumdy = saz*sel;
  sumdz = cel;
  used->data[z+(x+y*xsize)*zsize] = USED;
  /* try neighbors as new region points */

  //printf("\t\t Grow Loop...\n");
  //fflush(stdout);
  double grads_az,grads_el;

  //#pragma omp parallel default(none) shared(xsize,ysize,zsize,azimg,elimg,prec,vp_0,vp_1) private(xx,yy,zz,grads_az,grads_el,x_t,x_tminus,vp_x1,vp_x0,vp_11,vp_10)   
  //{
  //#pragma omp for reduction(+:vp_0) reduction(+:vp_1)
  //if(i<NOUT)
  //int coscount=0;
  for(i=0; i<*reg_size; i++)
    for(zz=reg[i].z-1; zz<=reg[i].z+1; zz++) 
    for(yy=reg[i].y-1; yy<=reg[i].y+1; yy++)
    for(xx=reg[i].x-1; xx<=reg[i].x+1; xx++)
       {
        grads_az = angles->az->data[ zz + zsize*(xx + yy * xsize) ]; 
        //grads_az = angles->az->data[ zz + zsize*(xx + yy * xsize) ]; 
        grads_el = angles->el->data[ zz + zsize*(xx + yy * xsize) ]; 
        //grads_el = angles->el->data[ zz + zsize*(xx + yy * xsize) ]; 
        //isaligned3(xx,yy,zz,angles,*reg_angle,prec) )
        if( xx>=0 && yy>=0 && zz>=0 && 
            xx<xsize && yy<ysize &&
            zz<zsize &&
            used->data[zz+(xx+yy*xsize)*zsize] != USED &&
            isaligned3(grads_az,grads_el,reg_az,reg_el,prec) )
          {
          
            //printf("\t\t\t Add points...\n");
            //fflush(stdout);
            /* add point */
            used->data[zz+(xx+yy*xsize)*zsize] = USED;
            reg[*reg_size].x = xx;
            reg[*reg_size].y = yy;
            reg[*reg_size].z = zz;
            ++(*reg_size);
 
            sincos(grads_az,&saz,&caz);
  	    sincos(grads_el,&sel,&cel);
            //printf("\t\t\t (%.2f,%.2f),(%.2f,%.2f)\n",grads_az*180./M_PI,grads_el*180./M_PI,reg_az*180./M_PI,reg_el*180./M_PI);
            //fflush(stdout);
            /* update region's angle */
            sumdx += caz*sel;
            sumdy += saz*sel;
            sumdz += cel;
	    //coscount++;
            reg_az = atan2(sumdy,sumdx);
            reg_el = acos(sumdz/sqrt(sumdx*sumdx + sumdy*sumdy + sumdz*sumdz));
            //sum is NOT necessarily normalized due to incrementation  
	    //calc_quat(&(*reg_angle)); 
            //printf("\t\t\t Updated!\n");
            //fflush(stdout);
          }}
	(*reg_angle)->az=reg_az;
	(*reg_angle)->el=reg_el;
	//printf("next\n\n\n");fflush(stdout);
  //printf("\t\tEnd Grow\n");
  //fflush(stdout);
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


/** Try some rectangles variations to improve NFA value. Only if the
    rectangle is not meaningful (i.e., log_nfa <= log_eps).
 */
static double rect3_improve( struct rect3 * rec, grads angles,
                            double logNT, double log_eps,double* mnfa,double* mnfa_2,double* mnfa_4,int Nnfa,int minsize, int minsize2,int minsize4 )
{
  struct rect3 r;
  double log_nfa,log_nfa_new;
  double delta = 0.5;
  double delta_2 = delta / 2.0;
  int n;
  //printf("\t %.0f,%.0f,%.0f",rec->dw1[0],rec->dw1[1],rec->dw1[2]);fflush(stdout);
  rect3_copy(rec,&r);
 if(r.p>0.1)
      log_nfa = rect3_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
 else if(r.p>0.05)
           log_nfa= rect3_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa = rect3_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
  

  if( log_nfa > log_eps ) return log_nfa;

  /* try finer precisions */
  rect3_copy(rec,&r);
  for(n=0; n<1; n++)
    {
      r.p /= 2.0;
      r.prec = r.p * M_PI;
      if(r.p>0.1)
      log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
      if( log_nfa_new > log_nfa )
        {
          log_nfa = log_nfa_new;
          rect3_copy(&r,rec);
        }
    }

  if( log_nfa > log_eps ) return log_nfa;
  /* try to reduce width */
  rect3_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width1 - delta) >= 0.5 )
        {
          r.width1 -= delta;
           if(r.p>0.1)
      log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
          
          if( log_nfa_new > log_nfa )
            {
              rect3_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }

      if( (r.width2 - delta) >= 0.5 )
        {
          r.width2 -= delta;
           if(r.p>0.1)
      log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
          
          if( log_nfa_new > log_nfa )
            {
              rect3_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try to reduce one side of the rectangle */
  rect3_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width1 - delta) >= 0.5 )
        {
          r.x1 +=  r.dw1[0] * delta_2;
          r.y1 +=  r.dw1[1] * delta_2;
          r.z1 +=  r.dw1[2] * delta_2;
      r.x2 +=  r.dw1[0] * delta_2;
          r.y2 +=  r.dw1[1] * delta_2;
          r.z2 +=  r.dw1[2] * delta_2;
      r.width1 -= delta;
           
         if(r.p>0.1)
      log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
         
          if( log_nfa_new > log_nfa )
            {
              rect3_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }

      if( (r.width2 - delta) >= 0.5 )
        {
          r.x1 +=  r.dw2[0] * delta_2;
          r.y1 +=  r.dw2[1] * delta_2;
          r.z1 +=  r.dw2[2] * delta_2;
      r.x2 +=  r.dw2[0] * delta_2;
          r.y2 +=  r.dw2[1] * delta_2;
          r.z2 +=  r.dw2[2] * delta_2;
      r.width2 -= delta;
           
         if(r.p>0.1)
      log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
         
          if( log_nfa_new > log_nfa )
            {
              rect3_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try to reduce the other side of the rectangle */
  rect3_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width1 - delta) >= 0.5 )
        {
          r.x1 -=  r.dw1[0] * delta_2;
          r.y1 -=  r.dw1[1] * delta_2;
          r.z1 -=  r.dw1[2] * delta_2;
      r.x2 -=  r.dw1[0] * delta_2;
          r.y2 -=  r.dw1[1] * delta_2;
          r.z2 -=  r.dw1[2] * delta_2;
      r.width1 -= delta;
           
         if(r.p>0.1)
      log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
         
          if( log_nfa_new > log_nfa )
            {
              rect3_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }

      if( (r.width2 - delta) >= 0.5 )
        {
          r.x1 -=  r.dw2[0] * delta_2;
          r.y1 -=  r.dw2[1] * delta_2;
          r.z1 -=  r.dw2[2] * delta_2;
      r.x2 -=  r.dw2[0] * delta_2;
          r.y2 -=  r.dw2[1] * delta_2;
          r.z2 -=  r.dw2[2] * delta_2;
      r.width2 -= delta;
           
         if(r.p>0.1)
      log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
         
          if( log_nfa_new > log_nfa )
            {
              rect3_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }
    }


  if( log_nfa > log_eps ) return log_nfa;

  /* try even finer precisions */
  rect3_copy(rec,&r);
  for(n=0; n<1; n++)
    {
      r.p /= 2.0;
      r.prec = r.p * M_PI;
      if(r.p>0.1)
      log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
      else if(r.p>0.05)
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
      else
           log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
      if( log_nfa_new > log_nfa )
        {
          log_nfa = log_nfa_new;
          rect3_copy(&r,rec);
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

/** Reduce the region size, by elimination the points far from the
    starting point, until that leads to rectangle with the right
    density of region points or to discard the region if too small.
 */
static int reduce_region3_radius( struct point3 * reg, int * reg_size,
                                 image3_double modgrad, angles3 reg_angle,
                                 double prec, double p, struct rect3 * rec,
                                 image3_char used, grads angles,
                                 double density_th )
{
  double density,rad1,rad2,rad,xc,yc,zc;
  int i;

  /* check parameters */
  if( reg == NULL ) error("reduce_region3_radius: invalid pointer 'reg'.");
  if( reg_size == NULL )
    error("reduce_region3_radius: invalid pointer 'reg_size'.");
  if( prec < 0.0 ) error("reduce_region3_radius: 'prec' must be positive.");
  if( rec == NULL ) error("reduce_region3_radius: invalid pointer 'rec'.");
  if( used == NULL || used->data == NULL )
    error("reduce_region3_radius: invalid image 'used'.");
  if( angles->az == NULL || angles->az->data == NULL )
    error("reduce_region3_radius: invalid image 'angles'.");

  /* compute region points density */
  /*require volume in denominator, in place of area*/
  density = (double) *reg_size /
                         ( dist3(rec->x1,rec->y1,rec->z1,rec->x2,rec->y2,rec->z2) 
               * rec->width1 * rec->width2 );

  /* if the density criterion is satisfied there is nothing to do */
  if( density >= density_th ) return TRUE;

  /* compute region's radius */
  /* def: distance from centerpoint to furthest endpoint*/
  xc = (double) reg[0].x;
  yc = (double) reg[0].y;
  zc = (double) reg[0].z;
  rad1 = dist3( xc, yc, zc, rec->x1, rec->y1, rec->z1 );
  rad2 = dist3( xc, yc, zc, rec->x2, rec->y2, rec->z2 );
  rad = rad1 > rad2 ? rad1 : rad2;

  /* while the density criterion is not satisfied, remove farther pixels */
  while( density < density_th )
    {
      rad *= 0.75; /* reduce region's radius to 75% of its value */

      /* remove points from the region and update 'used' map */
      for(i=0; i<*reg_size; i++)
        if( dist3( xc, yc, zc,  (double) reg[i].x, (double) reg[i].y, (double) reg[i].z ) > rad )
          {
            /* point not kept, mark it as NOTUSED */
            used->data[ reg[i].z + (reg[i].x + reg[i].y * used->xsize) * used->zsize ] = NOTUSED;
            /* remove point from the region */
            reg[i].x = reg[*reg_size-1].x; /* if i==*reg_size-1 copy itself */
            reg[i].y = reg[*reg_size-1].y;
        reg[i].z = reg[*reg_size-1].z;
            --(*reg_size);
            --i; /* to avoid skipping one point */
          }

      /* reject if the region is too small.
         2 is the minimal region size for 'region2rect' to work. */
      if( *reg_size < 2 ) return FALSE;

      /* re-compute rectangle */
      region2rect3(reg,*reg_size,modgrad,reg_angle,prec,p,rec);

      /* re-compute region points density */
      density = (double) *reg_size /
                         ( dist3(rec->x1,rec->y1,rec->z1,rec->x2,rec->y2,rec->z2) 
               * rec->width1 * rec->width2 );
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

/** Refine a rectangle.

    For that, an estimation of the angle tolerance is performed by the
    standard deviation of the angle at points near the region's
    starting point. Then, a new region is grown starting from the same
    point, but using the estimated angle tolerance. If this fails to
    produce a rectangle with the right density of region points,
    'reduce_region_radius' is called to try to satisfy this condition.
 */
static int refine3( struct point3 * reg, int * reg_size, image3_double modgrad,
                   angles3 reg_angle, double prec, double p, struct rect3 * rec,
                   image3_char used, grads angles,
		   double density_th , int NOUT)
{
  //double ang_d;
  double tau,density;//,xc,yc,zc;
  //double sum,s_sum,mean_angle;
  //angles3 ang_c, ang;
  int i;//,n;

  /* check parameters */
  if( reg == NULL ) error("refine3: invalid pointer 'reg'.");
  if( reg_size == NULL ) error("refine3: invalid pointer 'reg_size'.");
  if( prec < 0.0 ) error("refine3: 'prec' must be positive.");
  if( rec == NULL ) error("refine3: invalid pointer 'rec'.");
  if( used == NULL || used->data == NULL )
    error("refine3: invalid image 'used'.");
  if( angles ==NULL || angles->az == NULL || angles->az->data == NULL )
    error("refine3: invalid image 'angles'.");

  /* compute region points density */
  density = (double) *reg_size /
                     ( dist3(rec->x1,rec->y1,rec->z1,rec->x2,rec->y2,rec->z2) 
             * rec->width1 * rec->width2 );

  /* if the density criterion is satisfied there is nothing to do */
  if( density >= density_th ) return TRUE;

  /*------ First try: reduce angle tolerance ------*/

  /* compute the new mean angle and tolerance */
  /*
  xc = (double) reg[0].x;
  yc = (double) reg[0].y;
  zc = (double) reg[0].z;
  double az_c = angles->az->data[ reg[0].z + (reg[0].x + reg[0].y * angles->az->xsize) * angles->az->zsize ];
  double el_c = angles->el->data[ reg[0].z + (reg[0].x + reg[0].y * angles->el->xsize) * angles->el->zsize ];
  ang_c = new_angles3(az_c,el_c);
  */
  //Initialization of ang_d - to be overwritten 
  //ang = new_angles3(az_c,el_c);
  //sum = s_sum = 0.0;
  //n = 0;
  int xsize = (int) used->xsize;
  int zsize = (int) used->zsize;
  //scrub used memory for regrowing  
  for(i=0; i<*reg_size; i++)
    {
      used->data[ reg[i].z + (reg[i].x + reg[i].y * xsize) * zsize ] = NOTUSED;
    }  
  
	/*
 	if( dist3( xc, yc, zc, (double) reg[i].x, (double) reg[i].y, (double) reg[i].z) < min1(rec->width1,rec->width2) )
        {
          ang->az = angles->az->data[ reg[i].z + ( reg[i].x + reg[i].y * angles->az->xsize ) * angles->az->zsize ];
          ang->el = angles->el->data[ reg[i].z + ( reg[i].x + reg[i].y * angles->el->xsize ) * angles->el->zsize ];
          calc_quat(&ang);
      ang_d = quat_diff_signed(ang,ang_c);
          sum += ang_d;
      s_sum += ang_d * ang_d;
      ++n;
        }
    }
  mean_angle = sum / (double) n;
  tau = 2.0 * sqrt( (s_sum - 2.0 * mean_angle * sum) / (double) n
                         + mean_angle*mean_angle ); // 2 * standard deviation 
  */
  //free_angles3(ang_c);
  //free_angles3(ang);
  //NOTE - FIXED TAU IGNORES THIS COMPUTATION, BUR RUNNING FOR FUTURE USE  
  /* find a new region from the same starting point and new angle tolerance */
  
  tau=prec/2.0;
  region3_grow(reg[0].x,reg[0].y,reg[0].z,angles,reg,reg_size,&reg_angle,used,tau,NOUT);
/*prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;*/
  p=tau/M_PI;
  /* if the region is too small, reject */
  if( *reg_size < 2 ) return FALSE;

  /* re-compute rectangle */
  region2rect3(reg,*reg_size,modgrad,reg_angle,tau,p,rec);

  /* re-compute region points density */
  density = (double) *reg_size /
                     ( dist3(rec->x1,rec->y1,rec->z1,rec->x2,rec->y2,rec->z2) 
                 * rec->width1 * rec->width2 );
  /*------ Second try: reduce region radius ------*/
  if( density < density_th )
    return reduce_region3_radius( reg, reg_size, modgrad, reg_angle, prec, p,
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



static int isaligned3_markovV(double grads_az,double grads_el,double prec)
{
  if( grads_az == NOTDEF || grads_el == NOTDEF ) return FALSE;
  double diff = fabs(cos(grads_az)*sin(grads_el))   ;
  return (diff) <= sin(prec);
}
static int isaligned3_markovH(double grads_az,double grads_el,double prec)
{
  if( grads_az == NOTDEF || grads_el == NOTDEF ) return FALSE;
  double diff = fabs(sin(grads_az)*sin(grads_el))    ;
  return (diff) <= sin(prec);
}
static int isaligned3_markovD(double grads_az,double grads_el,double prec)
{
  if( grads_az == NOTDEF || grads_el == NOTDEF ) return FALSE;
  double diff =  fabs(cos(grads_el))   ;
  return (diff) <= sin(prec);
}
/** Given a conditioning image, estimate the Markov transition probabilities  for 
 *  Gradient-by-Ratio computation, P(1|1) and P(1|0).  
 */
static void make_markov3( double * img, int X, int Y, int Z,
                           double ang_th, int n_bins,
                           double * inputv)
{

  fprintf(stdout,"MakeMarkov3 started\n");
  fflush(stdout);

  //Instantiate variables per LSDSAR
  image3_double image;
  image3_double modgrad;
  grads angles; 
  struct coorlist3 * list_p;
  void * mem_p;
  unsigned int xsize,ysize,zsize;
  double prec,p;
  /* check parameters */
  if( img == NULL || X <= 0 || Y <= 0 || Z<= 0 ) error("invalid image input.");
  if( ang_th <= 0.0 || ang_th >= 180.0 ) error("'ang_th' value must be in the range (0,180).");
  if( n_bins <= 0 ) error("'n_bins' value must be positive.");
  /* angle tolerance */
  prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;
  double beta;
  beta=inputv[0];
  if(prec<0.0) error("MakeMarkov3: 'prec' must be positive");
  /* load and scale image (if necessary) and compute angle at each pixel */
  image = new_image3_double_ptr( (unsigned int) X, (unsigned int) Y, (unsigned int) Z, img );
  
  double start,end;
  start=omp_get_wtime();
  angles = ll_angle3( image, &list_p, &mem_p, &modgrad, (unsigned int) n_bins,beta);
  end=omp_get_wtime();
  //printf("LL_ANGLE: %f seconds\n",end-start);fflush(stdout);
  
  xsize = angles->az->xsize;
  ysize = angles->az->ysize;
  zsize = angles->az->zsize;

 //INDIVIDUAL LINE FREQUENCIES 
 //note that 'xx' is an index, while 'x' is the binary 0/1 alignment value for the kernel
 int ang_mult, xx, yy, zz;
 double hp_x0,hp_10,hp_x1,hp_11;
 double vp_x0,vp_10,vp_x1,vp_11;
 double dp_x0,dp_10,dp_x1,dp_11;
 double hp_1,vp_1,dp_1,hp_0,vp_0,dp_0;
 int x_t, x_tminus;
 angles3 angv,angh,angd;

 //angv = new_angles3(0.,M_PI/2.);
 //angh = new_angles3(M_PI/2.,M_PI/2.);
 //angd = new_angles3(0., 0.);
 /* dl={cos(az)sin(el),sin(az)sin(el),cos(el)}
  * hence el=pi/2 for the horizontal case (sin=1,cos=0)
  * and   el=0    for the vertical   case (sin=0,cos=1)*/
 angv = new_angles3(0.,M_PI/2.);
 angh = new_angles3(M_PI/2.,M_PI/2.);//.);
 angd = new_angles3(0.,0.);
 
 /* Calculate horizontal and vertical alignment freqencies*/
 // iterate precision options (prec, prec/2, prec/4)
/*
#pragma omp parallel
{
    printf("Hello World: Thread = %d\n",omp_get_thread_num());
    fflush(stdout);
} 
*/

double * azimg;
azimg = (double *) malloc(xsize*ysize*zsize*sizeof(double));
double * elimg; 
elimg = (double *) malloc(xsize*ysize*zsize*sizeof(double));
int i;
for(i=0;i<(xsize*ysize*zsize);i++)
{
    azimg[i]= angles->az->data[i];
    elimg[i]= angles->el->data[i];
}
double grads_az,grads_el;
for(ang_mult=5;ang_mult<=9;ang_mult+=2)
 {
 
     //printf("Make angle %d...",ang_mult);fflush(stdout);
     hp_1=0; hp_0=0; vp_1=0;  vp_0=0; dp_1=0; dp_0=0;
     
     // iterate vertical lines
     start=omp_get_wtime();
     
    #pragma omp parallel default(none) shared(xsize,ysize,zsize,azimg,elimg,prec,vp_0,vp_1) private(xx,yy,zz,grads_az,grads_el,x_t,x_tminus,vp_x1,vp_x0,vp_11,vp_10)   
    {
    #pragma omp for reduction(+:vp_0) reduction(+:vp_1)
    for(xx=0;xx<xsize;xx+=1)
     {
         //#pragma omp single    
         //{
         vp_x0 = 0; vp_10 = 0; vp_x1 = 0; vp_11 = 0;
         //}
         for(yy=0;yy<ysize-1;yy+=1)
         for(zz=0;zz<zsize-1;zz+=1)
         {
                
                grads_az = azimg[ (zz+1) + zsize*((xx+0) + (yy+1) * xsize) ]; 
                grads_el = elimg[ (zz+1) + zsize*((xx+0) + (yy+1) * xsize) ]; 
                //grads_az = angles->az->data[ (zz+1) + zsize*((xx+0) + (yy+1) * xsize) ]; 
                //grads_el = angles->el->data[ (zz+1) + zsize*((xx+0) + (yy+1) * xsize) ]; 
                x_t        = isaligned3_markovV(grads_az,grads_el,prec);
                
                grads_az = azimg[ (zz) + zsize*((xx) + (yy) * xsize) ]; 
                grads_el = elimg[ (zz) + zsize*((xx) + (yy) * xsize) ]; 
		//grads_az = angles->az->data[ zz + zsize*(xx + yy * xsize) ]; 
                //grads_el = angles->el->data[ zz + zsize*(xx + yy * xsize) ]; 
                x_tminus   = isaligned3_markovV(grads_az,grads_el,prec);
                
                if ( x_tminus==0 ) {++vp_x0; if ( x_t==1 ) ++vp_10;}
                else {++vp_x1; if ( x_t==1 ) ++vp_11;}
          }
          
          //#pragma omp single
          //{
          if( vp_x1>0) vp_1+=vp_11/vp_x1;
          vp_0+=vp_10/vp_x0;
          //}
      }
      }
      end=omp_get_wtime();
      //printf("Parallel: %f seconds\n",end-start);fflush(stdout);
     // iterate horizontal lines 
     start=omp_get_wtime();


    #pragma omp parallel default(none) shared(xsize,ysize,zsize,azimg,elimg,prec,hp_0,hp_1) private(xx,yy,zz,grads_az,grads_el,x_t,x_tminus,hp_x1,hp_x0,hp_11,hp_10)   
    {
    #pragma omp for reduction(+:hp_0) reduction(+:hp_1)
     for(yy=0;yy<ysize;yy+=1)
     {
         hp_x0 = 0; hp_10 = 0; hp_x1 = 0; hp_11 = 0;
         for(xx=0;xx<xsize-1;xx+=1)
             for(zz=0;zz<zsize-1;zz+=1)
         {
                

                grads_az = azimg[ (zz+1) + zsize*((xx+1) + (yy+0) * xsize) ]; 
                grads_el = elimg[ (zz+1) + zsize*((xx+1) + (yy+0) * xsize) ]; 
                //grads_az = angles->az->data[ (zz+1) + zsize*((xx+1) + (yy+0) * xsize) ]; 
                //grads_el = angles->el->data[ (zz+1) + zsize*((xx+1) + (yy+0) * xsize) ]; 
                x_t        = isaligned3_markovH(grads_az,grads_el,prec);
                grads_az = azimg[ (zz) + zsize*((xx) + (yy) * xsize) ]; 
                grads_el = elimg[ (zz) + zsize*((xx) + (yy) * xsize) ]; 
                //grads_az = angles->az->data[ zz + zsize*(xx + yy * xsize) ]; 
                //grads_el = angles->el->data[ zz + zsize*(xx + yy * xsize) ]; 
                x_tminus   = isaligned3_markovH(grads_az,grads_el,prec);
 
                if ( x_tminus==0 ){++hp_x0; if ( x_t==1 ) ++hp_10;}
                else {++hp_x1; if ( x_t==1 ) ++hp_11;}              
          }
          if( hp_x1>0) hp_1+=hp_11/hp_x1;
              hp_0+=hp_10/hp_x0;
      }
      }
      end=omp_get_wtime();
      //printf("Sequential: %f seconds\n",end-start);fflush(stdout);
     // iterate depth lines 
     start=omp_get_wtime();

    #pragma omp parallel default(none) shared(xsize,ysize,zsize,azimg,elimg,prec,dp_0,dp_1) private(xx,yy,zz,grads_az,grads_el,x_t,x_tminus,dp_x1,dp_x0,dp_11,dp_10)   
    {
    #pragma omp for reduction(+:dp_0) reduction(+:dp_1)
     for(zz=0;zz<zsize;zz+=1)
     {
         dp_x0 = 0; dp_10 = 0; dp_x1 = 0; dp_11 = 0;
         for(xx=0;xx<xsize-1;xx+=1)
             for(yy=0;yy<ysize-1;yy+=1)
         {
                
                grads_az=azimg[  (zz+0) + zsize*((xx+1) + (yy+1) * xsize)   ];
                grads_el=elimg[  (zz+0) + zsize*((xx+1) + (yy+1) * xsize)  ];
                //grads_az = angles->az->data[ (zz+0) + zsize*((xx+1) + (yy+1) * xsize) ]; 
                //grads_el = angles->el->data[ (zz+0) + zsize*((xx+1) + (yy+1) * xsize) ]; 
                x_t        = isaligned3_markovD(grads_az,grads_el,prec);         
                grads_az=azimg[  (zz) + zsize*((xx) + (yy) * xsize)   ];
                grads_el=elimg[  (zz) + zsize*((xx) + (yy) * xsize)  ];
                //grads_az = angles->az->data[ zz + zsize*(xx + yy * xsize) ]; 
                //grads_el = angles->el->data[ zz + zsize*(xx + yy * xsize) ]; 
                x_tminus   = isaligned3_markovD(grads_az,grads_el,prec);
                
                if ( x_tminus==0 ){++dp_x0; if ( x_t==1 ) ++dp_10;}
                else {++dp_x1; if ( x_t==1 ) ++dp_11;}              
          }
          if( dp_x1>0) dp_1+=dp_11/dp_x1;
              dp_0+=dp_10/dp_x0;
    }
    }
      end=omp_get_wtime();
      //printf("Memflops: %f seconds\n",end-start);fflush(stdout);
     //Catch extrema cases 
     inputv[ang_mult]   = (hp_1 + vp_1 + dp_1)/(xsize+ysize+zsize);
     if(inputv[ang_mult]<=0) inputv[ang_mult]=0.0001;
     if(inputv[ang_mult]>=1) inputv[ang_mult]=0.9999;
     inputv[ang_mult+1] = (hp_0 + vp_0 + dp_0)/(xsize+ysize+zsize);
     if(inputv[ang_mult+1]<=0) inputv[ang_mult+1]=0.0001;
     if(inputv[ang_mult+1]>=1) inputv[ang_mult+1]=0.9999;
     printf("prec %.2f\n",prec);
     printf("\t p11 : \t H: %.2f, V: %.2f, D: %.2f, HV: %.2f, HVD: %.2f\n",hp_1/xsize,vp_1/ysize,dp_1/zsize,(hp_1+vp_1)/(xsize+ysize),inputv[ang_mult]); 
     printf("\t p01 : \t H: %.2f, V: %.2f, D: %.2f, HV: %.2f, HVD: %.2f\n",hp_0/xsize,vp_0/ysize,dp_0/zsize,(hp_0+vp_0)/(xsize+ysize),inputv[ang_mult+1]); 
     fflush(stdout);
     // reduce tolerance for next loop 
     prec/=2;
     
 }
  
  free( (void *) image );   /* only the double_image structure should be freed,
                               the data pointer was provided to this functions
                              and should not be destroyed.                 */
  free_angles3(angh);
  free_angles3(angv);
  free_angles3(angd);
  free_grads(angles);
  //MAY REMOVE 
  free((void *)azimg);
  free((void *)elimg);
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
  //ang_th=ang_th/2.;
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
  double p01=inputv[6];
  double p00=1.0-p01;
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
/** LSD3 full interface.
 */
double * LineSegmentDetection3( int * n_out,
                               double * img, int X, int Y, int Z,
                               double ang_th, double log_eps, double density_th,
                               int n_bins,
                               int ** reg_img, int * reg_x, int * reg_y, int * reg_z ,double * inputv)
{
  
    
  //printf("Instantiating variables...\n");
  fflush(stdout);
    
  image3_double image;
  ntuple_list out = new_ntuple_list(10);
  double * return_value;
  grads angles;
  image3_double modgrad;
  image3_char used;
  image3_int region = NULL;
  struct coorlist3 * list_p;
  void * mem_p;
  struct rect3 rec;
  struct point3 * reg;
  int reg_size,min_reg_size,i;
  unsigned int xsize,ysize,zsize;
  angles3 reg_angle;
  double prec,p,log_nfa,logNT;
  int ls_count = 0;                   /* line segments are numbered 1,2,3,... */
  int ls_total = 0;

  //printf("Checking validity...\n");
  //fflush(stdout);
  
  /* check parameters */
  if( img == NULL || X <= 0 || Y <= 0 || Z<=0 ) error("invalid image3 input.");
  if( ang_th <= 0.0 || ang_th >= 180.0 )
    error("'ang_th' value must be in the range (0,180).");
  if( density_th < 0.0 || density_th > 1.0 )
    error("'density_th' value must be in the range [0,1].");
  if( n_bins <= 0 ) error("'n_bins' value must be positive.");


  /* angle tolerance */
  prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;
  // initialize angles3 for future edit
  reg_angle = new_angles3(0.,0.);
 
double beta;
beta=inputv[0];
int sizenum;
sizenum=(int) inputv[3];

    //printf("ImSize: %d\n",(int)sizenum);fflush(stdout);
    //for(i=0;i<=10;i++){printf("Input %d : %.4f\n",i,inputv[i]);fflush(stdout);}
  //printf("Creating image3...\n");
  //fflush(stdout);

  /* load and scale image (if necessary) and compute angle at each pixel */
  image = new_image3_double_ptr( (unsigned int) X, (unsigned int) Y, (unsigned int) Z,  img );
  
  //printf("Creating gradiants...\n");
  //fflush(stdout);

  angles = ll_angle3( image, &list_p, &mem_p, &modgrad,
                       (unsigned int) n_bins,beta);
  
  //printf("Accessing size...\n");
  //fflush(stdout);

  xsize = angles->az->xsize;
  ysize = angles->az->ysize;
  zsize = angles->az->zsize;
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
  
    In the 3D case, there are XYZ options for each endpoint.
    On the XY plane, there are sqrt(XY) options of the width1  
    we generalize and assume that (XYZ)^{1/3} options are considered 
    for both  width1 and width2, since we cannot isolate the dAz,dEl planes
    since there are (XY)^{1/2} width options in the 2d case
    Then we have Np*XYZ*XYZ*XYZ^{1/3}*XYZ^{1/3} =Np*(XYZ)^{8/3}
    Therefore, the runtime cost is higher by O[(XY)^{16/15}*Z^{8/3}]
    NT(3D)/NT(2D)=~O(Z^{8/3})
  */
 

  //printf("Saving region image...\n");
  //fflush(stdout);

   logNT = 8.0 * (log10((double)xsize) + log10((double)ysize) + log10((double)zsize)) / 3.0
          + log10(3.0);
  
   /* initialize some structures */
  if( reg_img != NULL && reg_x != NULL && reg_y != NULL && reg_z != NULL ) /* save region data */
    region = new_image3_int_ini(xsize,ysize,zsize,0);
  used = new_image3_char_ini(xsize,ysize,zsize,NOTUSED);
  reg = (struct point3 *) calloc( (size_t) (xsize*ysize*zsize), sizeof(struct point3) );
  if( reg == NULL ) error("not enough memory!");
double p0,p1;
  p1=p;
  p0=1-p1;
  //for(int inidx=5;inidx<10;inidx++) inputv[inidx]=inputv[inidx]/10.; 
  double p11=inputv[5];
 
  double p10=1.0-p11;
 
  
  double p01;
  p01=inputv[6];
 
  double p00;
  p00=1.0-p01;
   
                                   
 min_reg_size=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
 int N=sizenum;
 //printf("COnstructing NFA_matrix 1\n");fflush(stdout);
  double *output;
  output=( double *) malloc(N*N*sizeof(double));
  int NOUT=N;
  NFA_matrix(output,p0,p11,p01,N);

 //printf("COnstructing NFA_matrix 2\n");fflush(stdout);
  double *output_2;
  output_2=( double *) malloc(N*N*sizeof(double));
 p11=inputv[7];
 p01=inputv[8];
p1=p/2;
p0=1-p1;
int min_reg_size_2;
min_reg_size_2=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
NFA_matrix(output_2,p0,p11,p01,N);
  
 //printf("COnstructing NFA_matrix 3\n");fflush(stdout);
double *output_4;
  output_4=( double *) malloc(N*N*sizeof(double));
  p11=inputv[9];
  p01=inputv[10];
p1=p/4;
p0=1-p1;
int min_reg_size_4;
min_reg_size_4=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
NFA_matrix(output_4,p0,p11,p01,N);

 printf("All NFA matrices instantiated \n");fflush(stdout);

  printf("Searching for line segments...\n");
  fflush(stdout);
  int maxidx=0;
  int tempidx=0;
  /* search for line segments */
//int percentcount=0;
//int allcount=0;
//printf("Percent complete: ");fflush(stdout);

//double* az = (double*) angles->az->data;
//double* el = (double*) angles->el->data;

double * azimg;
azimg = (double *) malloc(xsize*ysize*zsize*sizeof(double));
double * elimg; 
elimg = (double *) malloc(xsize*ysize*zsize*sizeof(double));
for(i=0;i<(xsize*ysize*zsize);i++)
{
    azimg[i]= angles->az->data[i];
    elimg[i]= angles->el->data[i];
}
double startT,endT,growtime,regiontime,refinetime,improvetime;
int NOUT2=(int) xsize*ysize*zsize / 1;
for(; list_p != NULL; list_p = list_p->next )
{
    
    //printf("%d...",allcount);fflush(stdout);
    //if(allcount%1250000==0){printf("%d...",percentcount);fflush(stdout);percentcount++;}
    //allcount++;
    if( used->data[ list_p->z + (list_p->x + list_p->y *xsize) * zsize ] == NOTUSED &&
        azimg[ list_p->z + (list_p->x + list_p->y * xsize)*zsize ] != NOTDEF  &&
        elimg[ list_p->z + (list_p->x + list_p->y * xsize) *zsize ] != NOTDEF  )
       /* there is no risk of double comparison problems here
          because we are only interested in the exact NOTDEF value */
    {

    //printf("\t Growing...\n");
    //fflush(stdout);
        /* find the region of connected point and ~equal angle */
        
	startT=omp_get_wtime();
	region3_grow( list_p->x, list_p->y, list_p->z, angles, reg, &reg_size,
                     &reg_angle, used, prec, NOUT2);
	endT=omp_get_wtime();
	growtime=endT-startT;
    //printf("\t Reject small ...\n");
    //fflush(stdout);
        /* reject small regions */
        if( reg_size < min_reg_size ){ 
        ///printf("\t Regect %d",reg_size);fflush(stdout);
        continue;}

    //printf("\t Calc Rect...\n");
    //fflush(stdout);
        /* construct rectangular approximation for the region */

	startT=omp_get_wtime();
        region2rect3(reg,reg_size,modgrad,reg_angle,prec,p,&rec);
	endT=omp_get_wtime();
	regiontime=endT-startT;
        /* Check if the rectangle exceeds the minimal density of
           region points. If not, try to improve the region.
           The rectangle will be rejected if the final one does
           not fulfill the minimal density condition.
           This is an addition to the original LSD algorithm published in
           "LSD: A Fast Line Segment Detector with a False Detection Control"
           by R. Grompone von Gioi, J. Jakubowicz, J.M. Morel, and G. Randall.
           The original algorithm is obtained with density_th = 0.0.
         */

    //printf("\t Check refine...\n");
    //fflush(stdout);
     	startT=omp_get_wtime();
	 if( !refine3( reg, &reg_size, modgrad, reg_angle,
                     prec, p, &rec, used, angles,density_th,NOUT2 ) ) continue;
	endT=omp_get_wtime();
	refinetime=endT-startT;
	
        /* compute NFA value */ 
 

    //printf("\t Improve, get NFA, reg_size %d...\n",reg_size);
    //fflush(stdout);
        
	startT=omp_get_wtime();
	log_nfa = rect3_improve(&rec,angles,logNT,log_eps,output,output_2,output_4,NOUT,min_reg_size,min_reg_size_2,min_reg_size_4);
        endT=omp_get_wtime();
	improvetime=endT-startT;
        //if ( ls_total % 100 ==0 ){
    
    //printf("\t\t resolved %d, size %d...\n",ls_total, reg_size);
    //printf("\t\t resolved %d, size %d, z %.2f : %.2f...\n",ls_total, reg_size,rec.z1,rec.z2);
    //printf("\t\t\t (%.0f,%.0f,%.0f)-(%.0f,%.0f,%.0f) nfa %.2f, w(%.0f,%.0f)\n",rec.x1,rec.y1,rec.z1,rec.x2,rec.y2,rec.z2,log_nfa,rec.width1,rec.width2);
    //fflush(stdout);
        
        //++ls_total;
        if( log_nfa <= log_eps ) continue;
	printf("grow: %.4f, region: %.4f, refine: %.4f, improve: %.4f\n",growtime,regiontime,refinetime,improvetime);fflush(stdout);
        /* A New Line Segment was found! */
        ++ls_count;  /* increase line segment counter */

        /*
           The gradient was computed with a 2x2 mask, its value corresponds to
           points with an offset of (0.5,0.5), that should be added to output.
           The coordinates origin is at the center of pixel (0,0).
         */
        //rec.x1 += 0.; rec.y1 += 0.; rec.z1 += 0.;
        //rec.x2 += 0.; rec.y2 += 0.; rec.z2 += 0.;

      

        /* add line segment found to output */

        printf("\t LINE: NFA %.2f, lww: (%.2f, %.2f, %.2f), azel: (%.2f, %.2f)...\n",log_nfa,rec.length,rec.width1,rec.width2, rec.theta->az*180./M_PI, rec.theta->el*180./M_PI);
  	fflush(stdout);
        add_10tuple( out, rec.x1, rec.y1, rec.z1, rec.x2, rec.y2, rec.z2, 
                         rec.width1, rec.width2, rec.p, log_nfa );

        /* add region number to 'region' image if needed */
        if( region != NULL )
          for(i=0; i<reg_size; i++)
            region->data[ reg[i].z + (reg[i].x + reg[i].y * region->xsize) * region->zsize ] = ls_count;
      }
}
//printf("lscount %d, lstotal %d,  pts %.2f, used %.2f, az %.2f el %.2f, max %d\n",ls_count,ls_total,data0, data1/data0,data2/data0,data3/data0,maxidx);
//fflush(stdout);


  /* free memory */
  free( (void *) image );   /* only the double_image structure should be freed,
                               the data pointer was provided to this functions
                              and should not be destroyed.                 */
  free_grads(angles);
  free_angles3(reg_angle);
  free_image3_double(modgrad);
  free_image3_char(used);
  free( (void *) reg );
  free( (void *) mem_p );
free((void *) output); 
free((void *) output_2);
free((void *) output_4);
free(azimg);
free(elimg);
 //printf("Returning results...\n");fflush(stdout);


/* return the result */
  if( reg_img != NULL && reg_x != NULL && reg_y != NULL && reg_z != NULL)
    {
      if( region == NULL ) error("'region3' should be a valid image.");
      *reg_img = region->data;
      if( region->xsize > (unsigned int) INT_MAX ||
          region->ysize > (unsigned int) INT_MAX ||
      region->zsize > (unsigned int) INT_MAX)
        error("region image to big to fit in INT sizes.");
      *reg_x = (int) (region->xsize);
      *reg_y = (int) (region->ysize);
      *reg_z = (int) (region->zsize);
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
  //printf("Comlete! Returning...\n");
  //fflush(stdout);

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
 
  //printf("\n\n %.2f, %.2f, %.2f \n\n",beta,p1,p10);fflush(stdout);
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
//printf("\n\n%d\n\n",min_reg_size);fflush(stdout);
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
    //printf("%.2f\t",reg_angle);fflush(stdout);
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
        //printf("%.2f\t",log_nfa);fflush(stdout);    
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
        if(ndim==3)
        {
            printf("length - min: %.2f, max: %.2f\n",extrema[0],extrema[1]);
            printf("orient - min: %.2f, max: %.2f\n",extrema[2],extrema[3]);
            printf("width  - min: %.2f, max: %.2f\n",extrema[4],extrema[5]);
        }
        if(ndim==5)
        {
            printf("length - min: %.2f, max: %.2f\n",extrema[0],extrema[1]);
            printf("az     - min: %.2f, max: %.2f\n",extrema[2],extrema[3]);
            printf("el     - min: %.2f, max: %.2f\n",extrema[4],extrema[5]);
            printf("width1 - min: %.2f, max: %.2f\n",extrema[6],extrema[7]);
            printf("width2 - min: %.2f, max: %.2f\n",extrema[8],extrema[9]);
        }
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
    int sortsetting = ndim-1;
    if (ndim>4) sortsetting = ndim-2;
    while (sort_count<N){
        //get clustering center point, and fresh match list
        //sort_list[sort_count]=f0;sort_count++;
        for(i=0;i<(f0-1);i++){match[i]=0;}
        for(i=f0;i<nv;i++){match[i]=1;}
        //get matchlist for point
        for(param=0;param<sortsetting;param++)  {
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

double * CLSD3_Pipeline( int * n_out,
               double * img0, int X0, int Y0, int Z0,
               double * img, int X, int Y, int Z,
               double ang_th, double log_eps, double density_th, int n_bins, 
                           int ** reg_img, int * reg_x, int * reg_y, int*reg_z,
               double * inputv)
{
    int i,j,il;
    int mm=7;

    //Fix sizenum variable for noise image (l*5), 
    //1e5 rather than 1e4 for new dimension  
    double sizenum = pow(10.,3)*sqrt((double)X0*(double)X0 + 
                     (double)Y0*(double)Y0 + 
                 (double)Z0*(double)Z0);
    if(sizenum>pow(10.,6)) sizenum=pow(10.,6);
    printf("ImSize: %d",(int)sizenum);fflush(stdout);
    inputv[3] = sizenum;
    ang_th=ang_th;
    //n_bins*=2;
    //Caluculate Markov kernel
    double start,end;
    start=omp_get_wtime();
    make_markov3( img0, X0, Y0, Z0, ang_th, n_bins, inputv);
    end=omp_get_wtime();
    printf("MAKEMARKOV: %f seconds\n",end-start);fflush(stdout);
    //t = clock();
    printf("MARKOV COMPUTED\n");fflush(stdout);
    return LineSegmentDetection3( n_out, img, X, Y, Z,
                               ang_th, log_eps, density_th, n_bins,
                               reg_img, reg_x, reg_y, reg_z ,inputv);


    //t=clock()-t;
    //time_taken=((double)t)/CLOCKS_PER_SEC;
    //printf("LSD: %f seconds\n",time_taken);fflush(stdout);
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
        //printf("%d\t",sort_idxs[i]);fflush(stdout);
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
    //double mirsafety =1./(2.*card_factor); //higher than 1 for  lower-order approx 
    
    
    double mirsafety =1.5/pow(card_factor,.5); //higher than 1 for  lower-order approx 
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
    else {  return Conditional_LineSegmentDetection(n_out, img,X,Y,
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

/*----------------------------------------------------------------------------*/
/** LSD3 Simple Interface with Scale and Region output.
 */
double * lsd3_scale_region( int * n_out,
                           double * img, int X, int Y, int Z,
                           int ** reg_img, int * reg_x, int * reg_y, int * reg_z, double * inputv )
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

  return LineSegmentDetection3( n_out, img, X, Y, Z,
                               ang_th, log_eps, density_th, n_bins,
                               reg_img, reg_x, reg_y, reg_z ,inputv);
}

/*----------------------------------------------------------------------------*/
/** LSD Simple Interface with Scale.
 */
double * lsd3_scale(int * n_out, double * img, int X, int Y, int Z, double * inputv)
{
  return lsd3_scale_region(n_out,img,X,Y,Z,NULL,NULL,NULL,NULL,inputv);
}

/*----------------------------------------------------------------------------*/
/** LSD Simple Interface.
 */
double * lsd3(int * n_out, double * img, int X, int Y, int Z, double *inputv)
{
  /*fprintf(stderr,"LSD Error: %s\n","Just Testing");*/
  return lsd3_scale(n_out,img,X,Y,Z,inputv);
}

/*----------------------------------------------------------------------------*/
/** CLSD Simple Interface with Scale and Region output.
 */
double * c_lsd3_scale_region( int * n_out,
                             double * img, int X, int Y, int Z,
                 double * img0, int X0, int Y0, int Z0,
                             int ** reg_img, int * reg_x, int * reg_y, int * reg_z,
                 double * inputv )
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
  return CLSD3_Pipeline(n_out,
            img0, X0, Y0,Z0,
            img, X, Y,Z,
            ang_th, log_eps, density_th, n_bins,
                    reg_img, reg_x, reg_y , reg_z,inputv);
}


/*----------------------------------------------------------------------------*/
/** CLSD Simple Interface with Scale.
 */
double * c_lsd3_scale(int * n_out, 
                     double * img, int X, int Y, int Z,
                     double * img0, int X0, int Y0, int Z0,
                     double * inputv)
{
  return c_lsd3_scale_region(n_out,img,X,Y,Z,img0,X0,Y0,Z0,NULL,NULL,NULL,NULL,inputv);
}

/*----------------------------------------------------------------------------*/
/** CLSD Simple Interface.
 */
double * c_lsd3(int * n_out, 
               double * img, int X, int Y, int Z,
               double * img0, int X0, int Y0, int Z0,
               double *inputv)
{

  return c_lsd3_scale(n_out,img,X,Y,Z,img0,X0,Y0,Z0,inputv);
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
 
static PyObject * clsdmpi(PyObject * self, PyObject * args)
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
    int x,y,z,i,j,n;
    int X,Y,Z; 
    int X0,Y0,Z0;
    int Xin,Yin,Zin;
    int markovOnly;

    // parse arguments
    if (!PyArg_ParseTuple(args, "OIIIOIIIOI", &imagein, &X, &Y, &Z, &image0in, &X0, &Y0, &Z0, &inputvin, &markovOnly)) {
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
    
    if((markovOnly==1) & (Z<=1))
    {
        double ang_th;   /* Gradient angle tolerance in degrees.           */
        ang_th=inputv[4];
        int n_bins = 1024;       
        printf("Computing Markov kernel without further conditioning\n");
        fflush(stdout);
        if(Z<=1) 
        {
            make_markov(image0, X0, Y0, ang_th, n_bins,inputv);
            out = lsd(&n,image,X,Y,inputv);
        }
        else
        {
            make_markov3(image0, X0, Y0, Z0,  ang_th, n_bins,inputv);
            out = lsd3(&n,image,X,Y,Z,inputv);
        }
        printf("Kernel: (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f)\n",inputv[5],inputv[6], inputv[7],inputv[8],inputv[9],inputv[10]);
    fflush(stdout);
    } 
    else if(markovOnly==2)
    {
        double ang_th;   /* Gradient angle tolerance in degrees.           */
        ang_th=inputv[4];
        int n_bins = 1024;       
        printf("Computing Markov kernel and returning inputv\n");
        fflush(stdout);
        if(Z<=1) 
        {
            make_markov(image0, X0, Y0, ang_th, n_bins,inputv);
            out = inputv;
        }
        else
        {
            make_markov3(image0, X0, Y0, Z0,  ang_th, n_bins,inputv);
            out = inputv;
        }
        printf("Kernel: (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f)\n",inputv[5],inputv[6], inputv[7],inputv[8],inputv[9],inputv[10]);
    fflush(stdout);
    } 
    else if((X0==0) & (X>0))
    {
        printf("Using existing Markov kernel\n");
        fflush(stdout);
        if(Z<=1) out = lsd(&n,image,X,Y,inputv);
        else     out = lsd3(&n,image,X,Y,Z,inputv);
    }
    else
    {
        printf("Estimating Markov kernel\n");
        //printf("old: (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f) \n",inputv[5],inputv[6], inputv[7],inputv[8],inputv[9],inputv[10]);
        if(Z<=1) out =  c_lsd(&n,image,X,Y,image0,X0,Y0,inputv); 
        else     out = c_lsd3(&n,image,X,Y,Z,image0,X0,Y0,Z0,inputv);
        printf("Kernel: (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f)\n",inputv[5],inputv[6], inputv[7],inputv[8],inputv[9],inputv[10]);
        fflush(stdout);
    }
    printf("\n\nCOMPLETED RUN\n\n");fflush(stdout);
    
 
    // Convert output to a valid Python structure
    // accounting for various dimensionality options
    PyObject * pyout;
    if(markovOnly==2)
    {

        n_points = PySequence_Fast_GET_SIZE(inputvin);
    	pyout = PyList_New((int)n_points);
        if (!pyout) {
            return NULL;
        }   
        for (i = 0; i< (int)n_points; i++) 
        {
            PyObject *num = PyFloat_FromDouble(out[i]);
            if (!num) {
                Py_DECREF(pyout);
                return NULL;
            }
            PyList_SET_ITEM(pyout, i, num);
            
        }
    }
    else
    { 
        int mm=7;
        if(X==0) mm=4;
        else if(Z>1)  mm=10;
        pyout = PyList_New((int)n*mm);
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
    
    	free(out);
    }
   
    free(image);
    free(image0);
    free(inputv);
    free(outdata);
    //free(n_points);
    //free(x);free(y);free(z);free(i);free(j);free(n);
    //free(X);free(Y);free(Z);free(X0);free(Y0);free(Z0);
    //free(Xin);free(Yin);free(Zin);    
        
    Py_DECREF(imagein);
    Py_DECREF(image0in);
    Py_DECREF(inputvin);

    //free(imagein);    
    //free(image0in);
    //free(inputvin);
    //Py_DECREF(*num);
    return pyout; 
}

//Pythonic interfaces
static PyMethodDef clsdMethods[] = {
    {"clsdmpi", clsdmpi, METH_VARARGS, "Conditional variation of the LSDSAR algorithm"},
    {NULL, NULL, 0, NULL} /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "clsdmpi",
    NULL,
    -1,
    clsdMethods
};

PyMODINIT_FUNC PyInit_clsdmpi(void)
{
    return PyModule_Create(&moduledef);
}
