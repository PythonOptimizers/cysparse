#!/usr/bin/env python

# The file setup.py is automatically generated
# Generate it with
# python generate_code -s

from config.version import find_version, read
from config.config import get_path_option

from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension

import numpy as np

import ConfigParser
import os
import copy

from codecs import open
from os import path

###################################################################s####################################################
# HELPERS
########################################################################################################################
def prepare_Cython_extensions_as_C_extensions(extensions):
    """
    Modify the list of sources to transform `Cython` extensions into `C` extensions.

    Args:
        extensions: A list of (`Cython`) `distutils` extensions.

    Warning:
        The extensions are changed in place.

    Note:
        Only `Cython` source files are modified into their `C` equivalent source files. Other file types are unchanged.

    """
    for extension in extensions:
        c_sources = list()
        for source_path in extension.sources:
            path, source = os.path.split(source_path)
            filename, ext = os.path.splitext(source)

            if ext == '.pyx':
                c_sources.append(os.path.join(path, filename + '.c'))
            elif ext in ['.pxd', '.pxi']:
                pass
            else:
                # copy source as is
                c_sources.append(source_path)

        # modify extension in place
        extension.sources = c_sources

###################################################################s####################################################
# INIT
########################################################################################################################
cysparse_config_file = 'cysparse.cfg'
cysparse_config = ConfigParser.SafeConfigParser()
cysparse_config.read(cysparse_config_file)

numpy_include = np.get_include()


# Use Cython?
USE_CYTHON = cysparse_config.getboolean('CODE_GENERATION', 'USE_CYTHON')
if USE_CYTHON:
    try:
        from Cython.Distutils import build_ext
        from Cython.Build import cythonize
    except ImportError:
        raise ImportError("Check '%s': Cython is not properly installed." % cysparse_config_file)

# Debug mode?
DEBUG_SYMBOLS = cysparse_config.getboolean('CODE_GENERATION', 'DEBUG_SYMBOLS')

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

########################################################################################################################
# EXTENSIONS
########################################################################################################################
include_dirs = [numpy_include, '.']

ext_params = {}
ext_params['include_dirs'] = include_dirs
# -Wno-unused-function is potentially dangerous... use with care!
# '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION': doesn't work with Cython... because it **does** use a deprecated version...

if not DEBUG_SYMBOLS:
    ext_params['extra_compile_args'] = ["-O2", '-std=c99', '-Wno-unused-function']
    ext_params['extra_link_args'] = []
else:
    ext_params['extra_compile_args'] = ["-g", '-std=c99', '-Wno-unused-function']
    ext_params['extra_link_args'] = ["-g"]

########################################################################################################################
#                                                *** types ***
base_ext_params = copy.deepcopy(ext_params)
base_ext = [
    Extension(name="cysparse.cysparse_types.cysparse_types",
              sources=["cysparse/cysparse_types/cysparse_types.pxd", "cysparse/cysparse_types/cysparse_types.pyx"]),
    Extension(name="cysparse.cysparse_types.cysparse_numpy_types",
              sources=["cysparse/cysparse_types/cysparse_numpy_types.pxd", "cysparse/cysparse_types/cysparse_numpy_types.pyx"],
              **base_ext_params),
    Extension(name="cysparse.cysparse_types.cysparse_generic_types",
              sources=["cysparse/cysparse_types/cysparse_generic_types.pxd", "cysparse/cysparse_types/cysparse_generic_types.pyx"]),
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
    Extension("cysparse.utils.equality", ["cysparse/utils/equality.pxd",
                                          "cysparse/utils/equality.pyx"], **sparse_ext_params),
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

    # SPQR
    spqr_ext_params = copy.deepcopy(ext_params)
    print spqr_ext_params

    spqr_ext_params['include_dirs'].extend(suitesparse_include_dirs)
    spqr_ext_params['library_dirs'] = suitesparse_library_dirs
    spqr_ext_params['libraries'] = ['cholmod','spqr', 'amd']

    spqr_ext = [

  
        Extension(name="cysparse.linalg.suitesparse.spqr.spqr_INT32_t_FLOAT64_t",
                  sources=['cysparse/linalg/suitesparse/spqr/spqr_INT32_t_FLOAT64_t.pxd',
                           'cysparse/linalg/suitesparse/spqr/spqr_INT32_t_FLOAT64_t.pyx'], **spqr_ext_params),
    
        Extension(name="cysparse.linalg.suitesparse.spqr.spqr_INT32_t_COMPLEX128_t",
                  sources=['cysparse/linalg/suitesparse/spqr/spqr_INT32_t_COMPLEX128_t.pxd',
                           'cysparse/linalg/suitesparse/spqr/spqr_INT32_t_COMPLEX128_t.pyx'], **spqr_ext_params),
    

  
        Extension(name="cysparse.linalg.suitesparse.spqr.spqr_INT64_t_FLOAT64_t",
                  sources=['cysparse/linalg/suitesparse/spqr/spqr_INT64_t_FLOAT64_t.pxd',
                           'cysparse/linalg/suitesparse/spqr/spqr_INT64_t_FLOAT64_t.pyx'], **spqr_ext_params),
    
        Extension(name="cysparse.linalg.suitesparse.spqr.spqr_INT64_t_COMPLEX128_t",
                  sources=['cysparse/linalg/suitesparse/spqr/spqr_INT64_t_COMPLEX128_t.pxd',
                           'cysparse/linalg/suitesparse/spqr/spqr_INT64_t_COMPLEX128_t.pyx'], **spqr_ext_params),
    

        ]

########################################################################################################################
# config
########################################################################################################################
packages_list = ['cysparse',
            'cysparse.cysparse_types',
            'cysparse.sparse',
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
            #'cysparse.sparse.IO'
            'tests'
            ]

#packages_list=find_packages()

ext_modules = base_ext + sparse_ext + base_context_ext

if use_suitesparse:
    # add suitsparse package
    ext_modules += umfpack_ext
    ext_modules += cholmod_ext
    ext_modules += spqr_ext
    packages_list.append('cysparse.linalg.suitesparse')
    packages_list.append('cysparse.linalg.suitesparse.umfpack')
    packages_list.append('cysparse.linalg.suitesparse.cholmod')
    packages_list.append('cysparse.linalg.suitesparse.spqr')

########################################################################################################################
# PACKAGE PREPARATION FOR EXCLUSIVE C EXTENSIONS
########################################################################################################################
if not USE_CYTHON:
    prepare_Cython_extensions_as_C_extensions(ext_modules)

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

setup_args = {
    'name' :  'CySparse',
    'version' : find_version(os.path.realpath(__file__), 'cysparse', '__init__.py'),
    'description' : 'A Cython library for sparse matrices',
    'long_description' : long_description,
    #Author details
    'author' : 'Nikolaj van Omme, Sylvain Arreckx, Dominique Orban',

    'author_email' : 'cysparse\@TODO.com',

    'maintainer' : "CySparse Developers",

    'maintainer_email' : "dominique.orban@gerad.ca",

    'summary' : "Fast sparse matrix library for Python",
    'url' : "https://github.com/Funartech/cysparse",
    'download_url' : "https://github.com/Funartech/cysparse",
    'license' : 'LGPL',
    'classifiers' : filter(None, CLASSIFIERS.split('\n')),
    'install_requires' : ['numpy', 'Cython'],
    #ext_package' : 'cysparse', <- doesn't work with pxd files...
    #ext_modules = cythonize(ext_modules),
    'ext_modules' : ext_modules,
    'package_dir' : {"cysparse": "cysparse"},
    'packages' : packages_list,
    'zip_safe' : False

}

if USE_CYTHON:
    setup_args['cmdclass'] = {'build_ext': build_ext}

setup(**setup_args)




























