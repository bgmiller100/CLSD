# CLSD
Conditional Line Segment Detection on an observation image with regards to a noise model, assuming non-independent orientations in the null hypothesis

version 1.1 
October 9, 2020
Benjamin Miller
benjamin.g.miller@utexas.edu

#Requirements:
Python 3.7.6
MIR [1] : https://web.mit.edu/qiqi/www/mir/
GSL: https://www.gnu.org/software/gsl/

To compile, run 'bash setup.sh'.  

--

#NOTICE:
This program is modified from the source code of LSDSAR: [2]
Date of Modification: October 8, 2020.

#NOTICE: 
This program is modified from the source code of LSD: [3]
Date of Modification: 27/06/2018.

---

*[1] 
*"A high order multivariate approximation scheme for scattered datasets"
*by Qiqi Wang, Parviz Moin, and Gianluca Iaccarino.
*(Journal of COmputational Physics, 2010).
*https://doi.org/10.1016/j.jcp.2010.04.047

*[2]
*"LSDSAR, a Markovian a contrario framework for line segment detection in SAR images"
*by Chenguang Liu, Rémy Abergel, Yann Gousseau and Florence Tupin.
*(Pattern Recognition, 2019).
*https://doi.org/10.1016/j.patcog.2019.107034

*[3]
*"LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
*Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
*Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
*http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd



