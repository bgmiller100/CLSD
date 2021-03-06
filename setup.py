# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:31:11 2019
@author: Ben
"""

from distutils.core import setup, Extension 
import os
import sysconfig


#print(os.environ.get('TACC_GSL_LIB'))

d = os.getcwd()
exargs = sysconfig.get_config_var('CFLAGS').split()
exargs += ["-Wno-unused-variable","-Wno-unused-but-set-variable",
           "-Wno-unused-function","-Wno-sign-compare",
           "-Wno-maybe-uninitialized"]
module = Extension('clsd',
                    include_dirs = [os.environ.get('TACC_GSL_INC'), 
                                    '%s/mir_install/include'%d,
                                    '%s'%d],
                    libraries    = ['gsl', 'gslcblas', 'mir'],
                    library_dirs = [os.environ.get('TACC_GSL_LIB'),
                                    '%s/mir_install/lib'%d,
                                    '%s'%d],
                    runtime_library_dirs = [os.environ.get('TACC_GSL_LIB'),
                                    '%s/mir_install/lib'%d,
                                    '%s'%d],
                    sources = ['clsd.c'],
                    extra_compile_args = exargs)

setup(name = "clsd", ext_modules = [module])
