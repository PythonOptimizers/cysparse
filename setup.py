#!/usr/bin/env python

# The file setup.py is automatically generated
# Generate it with
# python generate_code -s

from config.version import find_version, read
from config.config import get_path_option

from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

import ConfigParser
import os
import copy

from codecs import open
from os import path

####################################################################s####################################################
# INIT
########################################################################################################################
cysparse_config = ConfigParser.SafeConfigParser()
cysparse_config.read('cysparse.cfg')

numpy_include = np.get_include()

# DEFAULT
default_include_dir = get_path_option(cysparse_config, 'DEFAULT', 'include_dirs')
default_library_dir = get_path_option(cysparse_config, 'DEFAULT', 'library_dirs')

# SUITESPARSE
# Do we use it or not?
use_suitesparse = cysparse_config.getboolean('SUITESPARSE', 'use_suitesparse')
# find user defined directories
if use_suitesparse:
    suitesparse_include_dirs = get_path_option(cysparse_config, 'SUITESPARSE', 'include_dirs')
    if suitesparse_include_dirs == '':
        suitesparse_include_dirs = default_include_dir
    suitesparse_library_dirs = get_path_option(cysparse_config, 'SUITESPARSE', 'library_dirs')
    if suitesparse_library_dirs == '':
        suitesparse_library_dirs = default_library_dir

# MUMPS
# Do we use it or not?
use_mumps = cysparse_config.getboolean('MUMPS', 'use_mumps')
mumps_compiled_in_64bits = cysparse_config.getboolean('MUMPS', 'mumps_compiled_in_64bits')

# find user defined directories
if use_mumps:
    mumps_include_dirs = get_path_option(cysparse_config, 'MUMPS', 'include_dirs')
    if mumps_include_dirs == '':
        mumps_include_dirs = default_include_dir
    mumps_library_dirs = get_path_option(cysparse_config, 'MUMPS', 'library_dirs')
    if mumps_library_dirs == '':
        mumps_library_dirs = default_library_dir

########################################################################################################################
# EXTENSIONS
########################################################################################################################
include_dirs = [numpy_include, '.']

ext_params = {}
ext_params['include_dirs'] = include_dirs
# -Wno-unused-function is potentially dangerous... use with care!
# '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION': doesn't work with Cython... because it **does** use a deprecated version...
ext_params['extra_compile_args'] = ["-O2", '-std=c99', '-Wno-unused-function']
ext_params['extra_link_args'] = []


########################################################################################################################
#                                                *** types ***
base_ext_params = copy.deepcopy(ext_params)
base_ext = [
    Extension(name="cysparse.types.cysparse_types",
              sources=["cysparse/types/cysparse_types.pxd", "cysparse/types/cysparse_types.pyx"]),
    Extension(name="cysparse.types.cysparse_numpy_types",
              sources=["cysparse/types/cysparse_numpy_types.pxd", "cysparse/types/cysparse_numpy_types.pyx"],
              **base_ext_params),
    Extension(name="cysparse.types.cysparse_generic_types",
              sources=["cysparse/types/cysparse_generic_types.pxd", "cysparse/types/cysparse_generic_types.pyx"]),
    ]

########################################################################################################################
#                                                *** sparse ***
sparse_ext_params = copy.deepcopy(ext_params)

sparse_ext = [
  ######################
  # ### Sparse ###
  ######################

  Extension(name="cysparse.sparse.sparse_utils.generic.generate_indices_INT32_t",
            sources=["cysparse/sparse/sparse_utils/generic/generate_indices_INT32_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/generate_indices_INT32_t.pyx"],
            **sparse_ext_params),
  Extension(name="cysparse.sparse.sparse_utils.generic.sort_indices_INT32_t",
            sources=["cysparse/sparse/sparse_utils/generic/sort_indices_INT32_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/sort_indices_INT32_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.generate_indices_INT64_t",
            sources=["cysparse/sparse/sparse_utils/generic/generate_indices_INT64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/generate_indices_INT64_t.pyx"],
            **sparse_ext_params),
  Extension(name="cysparse.sparse.sparse_utils.generic.sort_indices_INT64_t",
            sources=["cysparse/sparse/sparse_utils/generic/sort_indices_INT64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/sort_indices_INT64_t.pyx"],
            **sparse_ext_params),



  Extension(name="cysparse.sparse.sparse_utils.generic.print_INT32_t",
            sources=["cysparse/sparse/sparse_utils/generic/print_INT32_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/print_INT32_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.print_INT64_t",
            sources=["cysparse/sparse/sparse_utils/generic/print_INT64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/print_INT64_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.print_FLOAT32_t",
            sources=["cysparse/sparse/sparse_utils/generic/print_FLOAT32_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/print_FLOAT32_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.print_FLOAT64_t",
            sources=["cysparse/sparse/sparse_utils/generic/print_FLOAT64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/print_FLOAT64_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.print_FLOAT128_t",
            sources=["cysparse/sparse/sparse_utils/generic/print_FLOAT128_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/print_FLOAT128_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.print_COMPLEX64_t",
            sources=["cysparse/sparse/sparse_utils/generic/print_COMPLEX64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/print_COMPLEX64_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.print_COMPLEX128_t",
            sources=["cysparse/sparse/sparse_utils/generic/print_COMPLEX128_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/print_COMPLEX128_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.print_COMPLEX256_t",
            sources=["cysparse/sparse/sparse_utils/generic/print_COMPLEX256_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/print_COMPLEX256_t.pyx"],
            **sparse_ext_params),



    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT32_t_INT32_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT32_t_INT32_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT32_t_INT32_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT32_t_INT32_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_INT32_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_INT32_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT32_t_INT64_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT32_t_INT64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT32_t_INT64_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT32_t_INT64_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_INT64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_INT64_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT32_t_FLOAT32_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT32_t_FLOAT32_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT32_t_FLOAT32_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT32_t_FLOAT32_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_FLOAT32_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_FLOAT32_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT32_t_FLOAT64_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT32_t_FLOAT64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT32_t_FLOAT64_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT32_t_FLOAT64_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_FLOAT64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_FLOAT64_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT32_t_FLOAT128_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT32_t_FLOAT128_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT32_t_FLOAT128_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT32_t_FLOAT128_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_FLOAT128_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_FLOAT128_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT32_t_COMPLEX64_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT32_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT32_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT32_t_COMPLEX64_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT32_t_COMPLEX128_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT32_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT32_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT32_t_COMPLEX128_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT32_t_COMPLEX256_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT32_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT32_t_COMPLEX256_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT32_t_COMPLEX256_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT32_t_COMPLEX256_t.pyx"],
            **sparse_ext_params),
    

    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT64_t_INT32_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT64_t_INT32_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT64_t_INT32_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT64_t_INT32_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_INT32_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_INT32_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT64_t_INT64_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT64_t_INT64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT64_t_INT64_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT64_t_INT64_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_INT64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_INT64_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT64_t_FLOAT32_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT64_t_FLOAT32_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT64_t_FLOAT32_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT64_t_FLOAT32_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_FLOAT32_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_FLOAT32_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT64_t_FLOAT64_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT64_t_FLOAT64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT64_t_FLOAT64_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT64_t_FLOAT64_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_FLOAT64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_FLOAT64_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT64_t_FLOAT128_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT64_t_FLOAT128_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT64_t_FLOAT128_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT64_t_FLOAT128_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_FLOAT128_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_FLOAT128_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT64_t_COMPLEX64_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT64_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT64_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT64_t_COMPLEX64_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT64_t_COMPLEX128_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT64_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT64_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT64_t_COMPLEX128_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.sparse_utils.generic.find_INT64_t_COMPLEX256_t",
            sources=["cysparse/sparse/sparse_utils/generic/find_INT64_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_INT64_t_COMPLEX256_t.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_INT64_t_COMPLEX256_t",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_INT64_t_COMPLEX256_t.pyx"],
            **sparse_ext_params),
    




  ######################
  # ### SparseMatrix ###
  ######################
  Extension(name="cysparse.sparse.s_mat",
            sources=["cysparse/sparse/s_mat.pxd",
                     "cysparse/sparse/s_mat.pyx"],
            **sparse_ext_params),


    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT32_t_INT32_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT32_t_INT32_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT32_t_INT32_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT32_t_INT64_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT32_t_INT64_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT32_t_INT64_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT32_t_FLOAT32_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT32_t_FLOAT32_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT32_t_FLOAT32_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT32_t_FLOAT64_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT32_t_FLOAT64_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT32_t_FLOAT64_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT32_t_FLOAT128_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT32_t_FLOAT128_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT32_t_FLOAT128_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT32_t_COMPLEX64_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT32_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT32_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT32_t_COMPLEX128_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT32_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT32_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT32_t_COMPLEX256_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT32_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT32_t_COMPLEX256_t.pyx"],
            **sparse_ext_params),
    

    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT64_t_INT32_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT64_t_INT32_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT64_t_INT32_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT64_t_INT64_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT64_t_INT64_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT64_t_INT64_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT64_t_FLOAT32_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT64_t_FLOAT32_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT64_t_FLOAT32_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT64_t_FLOAT64_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT64_t_FLOAT64_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT64_t_FLOAT64_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT64_t_FLOAT128_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT64_t_FLOAT128_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT64_t_FLOAT128_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT64_t_COMPLEX64_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT64_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT64_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT64_t_COMPLEX128_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT64_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT64_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),
    
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_INT64_t_COMPLEX256_t",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_INT64_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_INT64_t_COMPLEX256_t.pyx"],
            **sparse_ext_params),
    


  ######################
  # ### LLSparseMatrix ###
  ######################
  Extension(name="cysparse.sparse.ll_mat",
            sources=["cysparse/sparse/ll_mat.pxd",
                     "cysparse/sparse/ll_mat.pyx"],
            **sparse_ext_params),

# TODO: add the possibility to **not** use tabu combinations...

  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_INT32_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_INT32_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_INT32_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT32_t_INT32_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT32_t_INT32_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT32_t_INT32_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT32_t_INT32_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_INT64_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_INT64_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_INT64_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT32_t_INT64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT32_t_INT64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT32_t_INT64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT32_t_INT64_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT32_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_FLOAT32_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_FLOAT32_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT32_t_FLOAT32_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT32_t_FLOAT32_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT32_t_FLOAT32_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT32_t_FLOAT32_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT64_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_FLOAT64_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_FLOAT64_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT32_t_FLOAT64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT32_t_FLOAT64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT32_t_FLOAT64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT32_t_FLOAT64_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT128_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_FLOAT128_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_FLOAT128_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT32_t_FLOAT128_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT32_t_FLOAT128_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT32_t_FLOAT128_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT32_t_FLOAT128_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX64_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_COMPLEX64_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT32_t_COMPLEX64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT32_t_COMPLEX64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT32_t_COMPLEX64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT32_t_COMPLEX64_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX128_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_COMPLEX128_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT32_t_COMPLEX128_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT32_t_COMPLEX128_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT32_t_COMPLEX128_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT32_t_COMPLEX128_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX256_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT32_t_COMPLEX256_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT32_t_COMPLEX256_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT32_t_COMPLEX256_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT32_t_COMPLEX256_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT32_t_COMPLEX256_t.pxi",
  
                     ],
            **sparse_ext_params),
  

  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_INT32_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_INT32_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_INT32_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT64_t_INT32_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT64_t_INT32_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT64_t_INT32_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT64_t_INT32_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_INT64_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_INT64_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_INT64_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT64_t_INT64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT64_t_INT64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT64_t_INT64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT64_t_INT64_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT32_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_FLOAT32_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_FLOAT32_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT64_t_FLOAT32_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT64_t_FLOAT32_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT64_t_FLOAT32_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT64_t_FLOAT32_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT64_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_FLOAT64_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_FLOAT64_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT64_t_FLOAT64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT64_t_FLOAT64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT64_t_FLOAT64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT64_t_FLOAT64_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT128_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_FLOAT128_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_FLOAT128_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT64_t_FLOAT128_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT64_t_FLOAT128_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT64_t_FLOAT128_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT64_t_FLOAT128_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX64_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_COMPLEX64_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT64_t_COMPLEX64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT64_t_COMPLEX64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT64_t_COMPLEX64_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT64_t_COMPLEX64_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX128_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_COMPLEX128_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT64_t_COMPLEX128_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT64_t_COMPLEX128_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT64_t_COMPLEX128_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT64_t_COMPLEX128_t.pxi",
  
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX256_t",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_INT64_t_COMPLEX256_t.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_INT64_t_COMPLEX256_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_INT64_t_COMPLEX256_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_INT64_t_COMPLEX256_t.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT64_t_COMPLEX256_t.pxi",
  
                     ],
            **sparse_ext_params),
  


  ######################
  # ### CSRSparseMatrix ###
  ######################

  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT32_t_INT32_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_INT32_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_INT32_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT32_t_INT32_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT32_t_INT32_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT32_t_INT64_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_INT64_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_INT64_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT32_t_INT64_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT32_t_INT64_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT32_t_FLOAT32_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_FLOAT32_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_FLOAT32_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT32_t_FLOAT32_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT32_t_FLOAT32_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT32_t_FLOAT64_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_FLOAT64_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_FLOAT64_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT32_t_FLOAT64_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT32_t_FLOAT64_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT32_t_FLOAT128_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_FLOAT128_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_FLOAT128_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT32_t_FLOAT128_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT32_t_FLOAT128_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT32_t_COMPLEX64_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_COMPLEX64_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT32_t_COMPLEX64_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT32_t_COMPLEX64_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT32_t_COMPLEX128_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_COMPLEX128_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT32_t_COMPLEX128_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT32_t_COMPLEX128_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT32_t_COMPLEX256_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT32_t_COMPLEX256_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT32_t_COMPLEX256_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT32_t_COMPLEX256_t.pxi",
                     ],
            **sparse_ext_params),
  

  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_INT32_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_INT32_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_INT32_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT64_t_INT32_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT64_t_INT32_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_INT64_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_INT64_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_INT64_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT64_t_INT64_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT64_t_INT64_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_FLOAT32_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_FLOAT32_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_FLOAT32_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT64_t_FLOAT32_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT64_t_FLOAT32_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_FLOAT64_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_FLOAT64_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_FLOAT64_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT64_t_FLOAT64_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT64_t_FLOAT64_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_FLOAT128_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_FLOAT128_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_FLOAT128_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT64_t_FLOAT128_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT64_t_FLOAT128_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_COMPLEX64_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_COMPLEX64_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT64_t_COMPLEX64_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT64_t_COMPLEX64_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_COMPLEX128_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_COMPLEX128_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT64_t_COMPLEX128_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT64_t_COMPLEX128_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_COMPLEX256_t",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_INT64_t_COMPLEX256_t.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_INT64_t_COMPLEX256_t.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT64_t_COMPLEX256_t.pxi",
                     ],
            **sparse_ext_params),
  


  ######################
  # ### CSCSparseMatrix ###
  ######################

  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_INT32_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_INT32_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_INT32_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT32_t_INT32_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT32_t_INT32_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_INT64_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_INT64_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_INT64_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT32_t_INT64_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT32_t_INT64_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_FLOAT32_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_FLOAT32_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_FLOAT32_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT32_t_FLOAT32_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT32_t_FLOAT32_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_FLOAT64_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_FLOAT64_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_FLOAT64_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT32_t_FLOAT64_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT32_t_FLOAT64_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_FLOAT128_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_FLOAT128_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_FLOAT128_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT32_t_FLOAT128_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT32_t_FLOAT128_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_COMPLEX64_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_COMPLEX64_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT32_t_COMPLEX64_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT32_t_COMPLEX64_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_COMPLEX128_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_COMPLEX128_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT32_t_COMPLEX128_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT32_t_COMPLEX128_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_COMPLEX256_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT32_t_COMPLEX256_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT32_t_COMPLEX256_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT32_t_COMPLEX256_t.pxi",
                     ],
            **sparse_ext_params),
  

  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_INT32_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_INT32_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_INT32_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT64_t_INT32_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT64_t_INT32_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_INT64_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_INT64_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_INT64_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT64_t_INT64_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT64_t_INT64_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_FLOAT32_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_FLOAT32_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_FLOAT32_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT64_t_FLOAT32_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT64_t_FLOAT32_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_FLOAT64_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_FLOAT64_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_FLOAT64_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT64_t_FLOAT64_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT64_t_FLOAT64_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_FLOAT128_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_FLOAT128_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_FLOAT128_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT64_t_FLOAT128_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT64_t_FLOAT128_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX64_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_COMPLEX64_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT64_t_COMPLEX64_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT64_t_COMPLEX64_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX128_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_COMPLEX128_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT64_t_COMPLEX128_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT64_t_COMPLEX128_t.pxi",
                     ],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX256_t",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_INT64_t_COMPLEX256_t.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_INT64_t_COMPLEX256_t.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT64_t_COMPLEX256_t.pxi",
                     ],
            **sparse_ext_params),
  


  ######################
  # ### LLSparseMatrixView ###
  ######################


  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_INT32_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_INT32_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_INT32_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_INT64_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_INT64_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_INT64_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_FLOAT32_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_FLOAT32_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_FLOAT32_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_FLOAT64_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_FLOAT64_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_FLOAT64_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_FLOAT128_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_FLOAT128_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_FLOAT128_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_COMPLEX64_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_COMPLEX128_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_COMPLEX256_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT32_t_COMPLEX256_t.pyx"],
            **sparse_ext_params),
  

  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT64_t_INT32_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_INT32_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_INT32_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT64_t_INT64_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_INT64_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_INT64_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT64_t_FLOAT32_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_FLOAT32_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_FLOAT32_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT64_t_FLOAT64_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_FLOAT64_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_FLOAT64_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT64_t_FLOAT128_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_FLOAT128_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_FLOAT128_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT64_t_COMPLEX64_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT64_t_COMPLEX128_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_INT64_t_COMPLEX256_t",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_INT64_t_COMPLEX256_t.pyx"],
            **sparse_ext_params),
  


  ######################
  # ### TransposedSparseMatrix ###
  ######################
  Extension(name="cysparse.sparse.sparse_proxies.t_mat",
            sources=["cysparse/sparse/sparse_proxies/t_mat.pxd",
                     "cysparse/sparse/sparse_proxies/t_mat.pyx"],
            **sparse_ext_params),
  ######################
  # ### ConjugateTransposedSparseMatrix ###
  ######################

  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT32_t_COMPLEX64_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT32_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT32_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT32_t_COMPLEX128_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT32_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT32_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT32_t_COMPLEX256_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT32_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT32_t_COMPLEX256_t.pyx"],
            **sparse_ext_params),
  

  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT64_t_COMPLEX64_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT64_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT64_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT64_t_COMPLEX128_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT64_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT64_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT64_t_COMPLEX256_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT64_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT64_t_COMPLEX256_t.pyx"],
            **sparse_ext_params),
  


  ######################
  # ### ConjugatedSparseMatrix ###
  ######################

  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT32_t_COMPLEX64_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT32_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT32_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT32_t_COMPLEX128_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT32_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT32_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT32_t_COMPLEX256_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT32_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT32_t_COMPLEX256_t.pyx"],
            **sparse_ext_params),
  

  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT64_t_COMPLEX64_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT64_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT64_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT64_t_COMPLEX128_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT64_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT64_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT64_t_COMPLEX256_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT64_t_COMPLEX256_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT64_t_COMPLEX256_t.pyx"],
            **sparse_ext_params),
  

]
########################################################################################################################
#                                                *** utils ***
utils_ext = [
    Extension("cysparse.utils.equality", ["cysparse/utils/equality.pxd", "cysparse/utils/equality.pyx"], **sparse_ext_params),
]

########################################################################################################################
#                                                *** LinAlg ***

##########################
# Base Contexts
##########################
context_ext_params = copy.deepcopy(ext_params)
base_context_ext = [

  
        Extension(name="cysparse.linalg.contexts.context_INT32_t_INT32_t",
                  sources=['cysparse/linalg/contexts/context_INT32_t_INT32_t.pxd',
                           'cysparse/linalg/contexts/context_INT32_t_INT32_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT32_t_INT64_t",
                  sources=['cysparse/linalg/contexts/context_INT32_t_INT64_t.pxd',
                           'cysparse/linalg/contexts/context_INT32_t_INT64_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT32_t_FLOAT32_t",
                  sources=['cysparse/linalg/contexts/context_INT32_t_FLOAT32_t.pxd',
                           'cysparse/linalg/contexts/context_INT32_t_FLOAT32_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT32_t_FLOAT64_t",
                  sources=['cysparse/linalg/contexts/context_INT32_t_FLOAT64_t.pxd',
                           'cysparse/linalg/contexts/context_INT32_t_FLOAT64_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT32_t_FLOAT128_t",
                  sources=['cysparse/linalg/contexts/context_INT32_t_FLOAT128_t.pxd',
                           'cysparse/linalg/contexts/context_INT32_t_FLOAT128_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT32_t_COMPLEX64_t",
                  sources=['cysparse/linalg/contexts/context_INT32_t_COMPLEX64_t.pxd',
                           'cysparse/linalg/contexts/context_INT32_t_COMPLEX64_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT32_t_COMPLEX128_t",
                  sources=['cysparse/linalg/contexts/context_INT32_t_COMPLEX128_t.pxd',
                           'cysparse/linalg/contexts/context_INT32_t_COMPLEX128_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT32_t_COMPLEX256_t",
                  sources=['cysparse/linalg/contexts/context_INT32_t_COMPLEX256_t.pxd',
                           'cysparse/linalg/contexts/context_INT32_t_COMPLEX256_t.pyx'], **context_ext_params),
    

  
        Extension(name="cysparse.linalg.contexts.context_INT64_t_INT32_t",
                  sources=['cysparse/linalg/contexts/context_INT64_t_INT32_t.pxd',
                           'cysparse/linalg/contexts/context_INT64_t_INT32_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT64_t_INT64_t",
                  sources=['cysparse/linalg/contexts/context_INT64_t_INT64_t.pxd',
                           'cysparse/linalg/contexts/context_INT64_t_INT64_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT64_t_FLOAT32_t",
                  sources=['cysparse/linalg/contexts/context_INT64_t_FLOAT32_t.pxd',
                           'cysparse/linalg/contexts/context_INT64_t_FLOAT32_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT64_t_FLOAT64_t",
                  sources=['cysparse/linalg/contexts/context_INT64_t_FLOAT64_t.pxd',
                           'cysparse/linalg/contexts/context_INT64_t_FLOAT64_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT64_t_FLOAT128_t",
                  sources=['cysparse/linalg/contexts/context_INT64_t_FLOAT128_t.pxd',
                           'cysparse/linalg/contexts/context_INT64_t_FLOAT128_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT64_t_COMPLEX64_t",
                  sources=['cysparse/linalg/contexts/context_INT64_t_COMPLEX64_t.pxd',
                           'cysparse/linalg/contexts/context_INT64_t_COMPLEX64_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT64_t_COMPLEX128_t",
                  sources=['cysparse/linalg/contexts/context_INT64_t_COMPLEX128_t.pxd',
                           'cysparse/linalg/contexts/context_INT64_t_COMPLEX128_t.pyx'], **context_ext_params),
    
        Extension(name="cysparse.linalg.contexts.context_INT64_t_COMPLEX256_t",
                  sources=['cysparse/linalg/contexts/context_INT64_t_COMPLEX256_t.pxd',
                           'cysparse/linalg/contexts/context_INT64_t_COMPLEX256_t.pyx'], **context_ext_params),
    


    ]
##########################
# SuiteSparse
##########################
if use_suitesparse:
    # UMFPACK
    umfpack_ext_params = copy.deepcopy(ext_params)
    umfpack_ext_params['include_dirs'].extend(suitesparse_include_dirs)
    umfpack_ext_params['library_dirs'] = suitesparse_library_dirs
    umfpack_ext_params['libraries'] = ['umfpack', 'amd']

    umfpack_ext = [

  
        Extension(name="cysparse.linalg.suitesparse.umfpack.umfpack_INT32_t_FLOAT64_t",
                  sources=['cysparse/linalg/suitesparse/umfpack/umfpack_INT32_t_FLOAT64_t.pxd',
                           'cysparse/linalg/suitesparse/umfpack/umfpack_INT32_t_FLOAT64_t.pyx'], **umfpack_ext_params),
    
        Extension(name="cysparse.linalg.suitesparse.umfpack.umfpack_INT32_t_COMPLEX128_t",
                  sources=['cysparse/linalg/suitesparse/umfpack/umfpack_INT32_t_COMPLEX128_t.pxd',
                           'cysparse/linalg/suitesparse/umfpack/umfpack_INT32_t_COMPLEX128_t.pyx'], **umfpack_ext_params),
    

  
        Extension(name="cysparse.linalg.suitesparse.umfpack.umfpack_INT64_t_FLOAT64_t",
                  sources=['cysparse/linalg/suitesparse/umfpack/umfpack_INT64_t_FLOAT64_t.pxd',
                           'cysparse/linalg/suitesparse/umfpack/umfpack_INT64_t_FLOAT64_t.pyx'], **umfpack_ext_params),
    
        Extension(name="cysparse.linalg.suitesparse.umfpack.umfpack_INT64_t_COMPLEX128_t",
                  sources=['cysparse/linalg/suitesparse/umfpack/umfpack_INT64_t_COMPLEX128_t.pxd',
                           'cysparse/linalg/suitesparse/umfpack/umfpack_INT64_t_COMPLEX128_t.pyx'], **umfpack_ext_params),
    

        ]

    # CHOLMOD
    cholmod_ext_params = copy.deepcopy(ext_params)
    print cholmod_ext_params

    cholmod_ext_params['include_dirs'].extend(suitesparse_include_dirs)
    cholmod_ext_params['library_dirs'] = suitesparse_library_dirs
    cholmod_ext_params['libraries'] = ['cholmod', 'amd']

    print cholmod_ext_params

    cholmod_ext = [

  
        Extension(name="cysparse.linalg.suitesparse.cholmod.cholmod_INT32_t_FLOAT64_t",
                  sources=['cysparse/linalg/suitesparse/cholmod/cholmod_INT32_t_FLOAT64_t.pxd',
                           'cysparse/linalg/suitesparse/cholmod/cholmod_INT32_t_FLOAT64_t.pyx'], **cholmod_ext_params),
    
        Extension(name="cysparse.linalg.suitesparse.cholmod.cholmod_INT32_t_COMPLEX128_t",
                  sources=['cysparse/linalg/suitesparse/cholmod/cholmod_INT32_t_COMPLEX128_t.pxd',
                           'cysparse/linalg/suitesparse/cholmod/cholmod_INT32_t_COMPLEX128_t.pyx'], **cholmod_ext_params),
    

  
        Extension(name="cysparse.linalg.suitesparse.cholmod.cholmod_INT64_t_FLOAT64_t",
                  sources=['cysparse/linalg/suitesparse/cholmod/cholmod_INT64_t_FLOAT64_t.pxd',
                           'cysparse/linalg/suitesparse/cholmod/cholmod_INT64_t_FLOAT64_t.pyx'], **cholmod_ext_params),
    
        Extension(name="cysparse.linalg.suitesparse.cholmod.cholmod_INT64_t_COMPLEX128_t",
                  sources=['cysparse/linalg/suitesparse/cholmod/cholmod_INT64_t_COMPLEX128_t.pxd',
                           'cysparse/linalg/suitesparse/cholmod/cholmod_INT64_t_COMPLEX128_t.pyx'], **cholmod_ext_params),
    

        ]

##########################
# MUMPS
##########################
if use_mumps:
    mumps_ext = []

  
    mumps_ext_params_INT32_t_FLOAT32_t = copy.deepcopy(ext_params)
    mumps_ext_params_INT32_t_FLOAT32_t['include_dirs'].extend(mumps_include_dirs)
    mumps_ext_params_INT32_t_FLOAT32_t['include_dirs'].append("/Users/syarra/work/VirtualEnvs/nlpy_new/programs/MUMPS.py/")
    mumps_ext_params_INT32_t_FLOAT32_t['library_dirs'] = mumps_library_dirs
    mumps_ext_params_INT32_t_FLOAT32_t['libraries'] = [] # 'scalapack', 'pord']
    mumps_ext_params_INT32_t_FLOAT32_t['libraries'].append('smumps')
    mumps_ext_params_INT32_t_FLOAT32_t['libraries'].append('mumps_common')
    mumps_ext_params_INT32_t_FLOAT32_t['libraries'].append('pord')
    mumps_ext_params_INT32_t_FLOAT32_t['libraries'].append('mpiseq')
    mumps_ext_params_INT32_t_FLOAT32_t['libraries'].append('blas')
    mumps_ext_params_INT32_t_FLOAT32_t['libraries'].append('pthread')

    mumps_ext.append(

        Extension(name="cysparse.linalg.mumps.mumps_INT32_t_FLOAT32_t",
                  sources=['cysparse/linalg/mumps/mumps_INT32_t_FLOAT32_t.pxd',
                           'cysparse/linalg/mumps/mumps_INT32_t_FLOAT32_t.pyx'], **mumps_ext_params_INT32_t_FLOAT32_t))
  
    mumps_ext_params_INT32_t_FLOAT64_t = copy.deepcopy(ext_params)
    mumps_ext_params_INT32_t_FLOAT64_t['include_dirs'].extend(mumps_include_dirs)
    mumps_ext_params_INT32_t_FLOAT64_t['include_dirs'].append("/Users/syarra/work/VirtualEnvs/nlpy_new/programs/MUMPS.py/")
    mumps_ext_params_INT32_t_FLOAT64_t['library_dirs'] = mumps_library_dirs
    mumps_ext_params_INT32_t_FLOAT64_t['library_dirs'].append("/Users/syarra/work/VirtualEnvs/nlpy_new/lib/python2.7/site-packages/MUMPS.py-0.1.0.dev0-py2.7-macosx-10.10-x86_64.egg/")
    mumps_ext_params_INT32_t_FLOAT64_t['libraries'] = [] # 'scalapack', 'pord']
    mumps_ext_params_INT32_t_FLOAT64_t['libraries'].append('dmumps')
    mumps_ext_params_INT32_t_FLOAT64_t['libraries'].append('mumps_common')
    mumps_ext_params_INT32_t_FLOAT64_t['libraries'].append('pord')
    mumps_ext_params_INT32_t_FLOAT64_t['libraries'].append('mpiseq')
    mumps_ext_params_INT32_t_FLOAT64_t['libraries'].append('blas')
    mumps_ext_params_INT32_t_FLOAT64_t['libraries'].append('pthread')

    mumps_ext.append(

        Extension(name="cysparse.linalg.mumps.mumps_INT32_t_FLOAT64_t",
                  sources=['cysparse/linalg/mumps/mumps_INT32_t_FLOAT64_t.pxd',
                           'cysparse/linalg/mumps/mumps_INT32_t_FLOAT64_t.pyx'], **mumps_ext_params_INT32_t_FLOAT64_t))
  
    mumps_ext_params_INT32_t_COMPLEX64_t = copy.deepcopy(ext_params)
    mumps_ext_params_INT32_t_COMPLEX64_t['include_dirs'].extend(mumps_include_dirs)
    mumps_ext_params_INT32_t_COMPLEX64_t['include_dirs'].append("/Users/syarra/work/VirtualEnvs/nlpy_new/programs/MUMPS.py/")
    mumps_ext_params_INT32_t_COMPLEX64_t['library_dirs'] = mumps_library_dirs
    mumps_ext_params_INT32_t_COMPLEX64_t['libraries'] = [] # 'scalapack', 'pord']
    mumps_ext_params_INT32_t_COMPLEX64_t['libraries'].append('cmumps')
    mumps_ext_params_INT32_t_COMPLEX64_t['libraries'].append('mumps_common')
    mumps_ext_params_INT32_t_COMPLEX64_t['libraries'].append('pord')
    mumps_ext_params_INT32_t_COMPLEX64_t['libraries'].append('mpiseq')
    mumps_ext_params_INT32_t_COMPLEX64_t['libraries'].append('blas')
    mumps_ext_params_INT32_t_COMPLEX64_t['libraries'].append('pthread')

    mumps_ext.append(

        Extension(name="cysparse.linalg.mumps.mumps_INT32_t_COMPLEX64_t",
                  sources=['cysparse/linalg/mumps/mumps_INT32_t_COMPLEX64_t.pxd',
                           'cysparse/linalg/mumps/mumps_INT32_t_COMPLEX64_t.pyx'], **mumps_ext_params_INT32_t_COMPLEX64_t))
  
    mumps_ext_params_INT32_t_COMPLEX128_t = copy.deepcopy(ext_params)
    mumps_ext_params_INT32_t_COMPLEX128_t['include_dirs'].extend(mumps_include_dirs)
    mumps_ext_params_INT32_t_COMPLEX128_t['include_dirs'].append("/Users/syarra/work/VirtualEnvs/nlpy_new/programs/MUMPS.py/")
    mumps_ext_params_INT32_t_COMPLEX128_t['library_dirs'] = mumps_library_dirs
    mumps_ext_params_INT32_t_COMPLEX128_t['libraries'] = [] # 'scalapack', 'pord']
    mumps_ext_params_INT32_t_COMPLEX128_t['libraries'].append('zmumps')
    mumps_ext_params_INT32_t_COMPLEX128_t['libraries'].append('mumps_common')
    mumps_ext_params_INT32_t_COMPLEX128_t['libraries'].append('pord')
    mumps_ext_params_INT32_t_COMPLEX128_t['libraries'].append('mpiseq')
    mumps_ext_params_INT32_t_COMPLEX128_t['libraries'].append('blas')
    mumps_ext_params_INT32_t_COMPLEX128_t['libraries'].append('pthread')

    mumps_ext.append(

        Extension(name="cysparse.linalg.mumps.mumps_INT32_t_COMPLEX128_t",
                  sources=['cysparse/linalg/mumps/mumps_INT32_t_COMPLEX128_t.pxd',
                           'cysparse/linalg/mumps/mumps_INT32_t_COMPLEX128_t.pyx'], **mumps_ext_params_INT32_t_COMPLEX128_t))
  



########################################################################################################################
# config
########################################################################################################################
packages_list = ['cysparse',
            'cysparse.types',
            'cysparse.sparse',
            'cysparse.sparse.like',
            'cysparse.sparse.sparse_proxies',
            'cysparse.sparse.sparse_proxies.complex_generic',
            'cysparse.sparse.sparse_utils',
            'cysparse.sparse.sparse_utils.generic',
            'cysparse.sparse.s_mat_matrices',
            'cysparse.sparse.ll_mat_matrices',
            'cysparse.sparse.csr_mat_matrices',
            'cysparse.sparse.csc_mat_matrices',
            'cysparse.sparse.ll_mat_views',
            'cysparse.utils',
            'cysparse.linalg',
            'cysparse.linalg.contexts',
            #'cysparse.linalg.mumps',
            #'cysparse.sparse.IO'
            'tests'
            ]

#packages_list=find_packages()

ext_modules = base_ext + sparse_ext + base_context_ext

if use_suitesparse:
    # add suitsparse package
    ext_modules += umfpack_ext
    ext_modules += cholmod_ext
    packages_list.append('cysparse.linalg.suitesparse')
    packages_list.append('cysparse.linalg.suitesparse.umfpack')
    packages_list.append('cysparse.linalg.suitesparse.cholmod')

if use_mumps:
    # add mumps
    ext_modules += mumps_ext
    packages_list.append('cysparse.linalg.mumps')

########################################################################################################################
# PACKAGE SPECIFICATIONS
########################################################################################################################

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Cython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS :: MacOS X
Natural Language :: English
"""

here = path.abspath(path.dirname(__file__))
# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name=  'CySparse',
      version=find_version(os.path.realpath(__file__), 'cysparse', '__init__.py'),
      description='A Cython library for sparse matrices',
      long_description=long_description,
      #Author details
      author='Nikolaj van Omme, Sylvain Arreckx, Dominique Orban',

      author_email='cysparse\@TODO.com',

      maintainer = "CySparse Developers",

      maintainer_email = "dominique.orban@gerad.ca",

      summary = "Fast sparse matrix library for Python",
      url = "https://github.com/Funartech/cysparse",
      download_url = "https://github.com/Funartech/cysparse",
      license='LGPL',
      classifiers=filter(None, CLASSIFIERS.split('\n')),
      install_requires=['numpy', 'Cython'],
      #ext_package='cysparse', <- doesn't work with pxd files...
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      package_dir = {"cysparse": "cysparse"},
      packages=packages_list,
      zip_safe=False
)
