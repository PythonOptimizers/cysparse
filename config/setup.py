#!/usr/bin/env python

########################################################################################################################
#                                                                                                                      #
#                              The file `setup.py` is automatically generated from `config/setup.cpy`                  #
#                                                                                                                      #
########################################################################################################################

from config.version import find_version, read
from config.config import get_path_option

from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension
from distutils import sysconfig

import numpy as np

import configparser
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
        The extensions are changed in place. This function is not compatible with `C++` code.

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
cysparse_config = configparser.SafeConfigParser()
cysparse_config.read(cysparse_config_file)

numpy_include = np.get_include()

# Use Cython?
use_cython = cysparse_config.getboolean('CODE_GENERATION', 'use_cython')
if use_cython:
    try:
        from Cython.Distutils import build_ext
        from Cython.Build import cythonize
    except ImportError:
        raise ImportError("Check '%s': Cython is not properly installed." % cysparse_config_file)

# Debug mode?
use_debug_symbols = cysparse_config.getboolean('CODE_GENERATION', 'use_debug_symbols')
use_compiler_optimization = cysparse_config.getboolean('CODE_GENERATION', 'use_compiler_optimization')

########################################################################################################################
# EXTENSIONS
########################################################################################################################
include_dirs = [numpy_include, '.']

ext_params = {}
ext_params['include_dirs'] = include_dirs
# -Wno-unused-function is potentially dangerous... use with care!
# '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION': doesn't work with Cython... because it **does** use a deprecated version...

ext_params['extra_compile_args'] = ['-std=c99', '-Wno-unused-function']
ext_params['extra_link_args'] = []

if not use_debug_symbols:
    key_to_modify = 'PY_CORE_CFLAGS'
    cflags = sysconfig._config_vars[key_to_modify]
    cflags = cflags.replace(' -g ', ' ')
    sysconfig._config_vars[key_to_modify] = cflags
else:
    ext_params['extra_compile_args'].append("-g")
    ext_params['extra_link_args'].append("-g")

if use_compiler_optimization:
    ext_params['extra_compile_args'].append("-O3")
else:
    ext_params['extra_compile_args'].append("-O2")


#-----------------------------------------------------------------------------------------------------------------------
#                                                *** types ***
base_ext_params = copy.deepcopy(ext_params)
base_ext = [
    Extension(name="cysparse.common_types.cysparse_types",
              sources=["cysparse/common_types/cysparse_types.pxd", "cysparse/common_types/cysparse_types.pyx"],
              **base_ext_params),
    Extension(name="cysparse.common_types.cysparse_numpy_types",
              sources=["cysparse/common_types/cysparse_numpy_types.pxd", "cysparse/common_types/cysparse_numpy_types.pyx"],
              **base_ext_params),
    Extension(name="cysparse.common_types.cysparse_generic_types",
              sources=["cysparse/common_types/cysparse_generic_types.pxd", "cysparse/common_types/cysparse_generic_types.pyx"],
              **base_ext_params),
    ]

#-----------------------------------------------------------------------------------------------------------------------
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
  

  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT64_t_COMPLEX64_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT64_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT64_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT64_t_COMPLEX128_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT64_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/h_mat_INT64_t_COMPLEX128_t.pyx"],
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
  

  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT64_t_COMPLEX64_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT64_t_COMPLEX64_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT64_t_COMPLEX64_t.pyx"],
            **sparse_ext_params),
  
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT64_t_COMPLEX128_t",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT64_t_COMPLEX128_t.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/conj_mat_INT64_t_COMPLEX128_t.pyx"],
            **sparse_ext_params),
  


  ######################
  # ### OpProxy ###
  ######################
  Extension(name="cysparse.sparse.operator_proxies.op_proxy",
            sources=["cysparse/sparse/operator_proxies/op_proxy.pxd",
                     "cysparse/sparse/operator_proxies/op_proxy.pyx"],
            **sparse_ext_params),

  ######################
  # ### OpScalarProxy ###
  ######################
  Extension(name="cysparse.sparse.operator_proxies.op_scalar_proxy",
            sources=["cysparse/sparse/operator_proxies/op_scalar_proxy.pxd",
                     "cysparse/sparse/operator_proxies/op_scalar_proxy.pyx"],
            **sparse_ext_params),

  ######################
  # ### ScalarMulProxy ###
  ######################
  Extension(name="cysparse.sparse.operator_proxies.scalar_mul_proxy",
            sources=["cysparse/sparse/operator_proxies/scalar_mul_proxy.pxd",
                     "cysparse/sparse/operator_proxies/scalar_mul_proxy.pyx"],
            **sparse_ext_params),

  ######################
  # ### OpMatrixProxy ###
  ######################
  Extension(name="cysparse.sparse.operator_proxies.op_matrix_proxy",
            sources=["cysparse/sparse/operator_proxies/op_matrix_proxy.pxd",
                     "cysparse/sparse/operator_proxies/op_matrix_proxy.pyx"],
            **sparse_ext_params),

  ######################
  # ### SumProxy ###
  ######################
  Extension(name="cysparse.sparse.operator_proxies.sum_proxy",
            sources=["cysparse/sparse/operator_proxies/sum_proxy.pxd",
                     "cysparse/sparse/operator_proxies/sum_proxy.pyx"],
            **sparse_ext_params),

  ######################
  # ### MulProxy ###
  ######################
  Extension(name="cysparse.sparse.operator_proxies.mul_proxy",
            sources=["cysparse/sparse/operator_proxies/mul_proxy.pxd",
                     "cysparse/sparse/operator_proxies/mul_proxy.pyx"],
            **sparse_ext_params),

]

#-----------------------------------------------------------------------------------------------------------------------
#                                                *** utils ***
utils_ext = [
    Extension("cysparse.utils.equality", ["cysparse/utils/equality.pxd",
                                          "cysparse/utils/equality.pyx"], **sparse_ext_params),
]

########################################################################################################################
# config
########################################################################################################################
packages_list = ['cysparse',
            'cysparse.common_types',
            'cysparse.sparse',
            'cysparse.sparse.sparse_proxies',
            'cysparse.sparse.operator_proxies',
            'cysparse.sparse.sparse_proxies.complex_generic',
            'cysparse.sparse.sparse_utils',
            'cysparse.sparse.sparse_utils.generic',
            'cysparse.sparse.s_mat_matrices',
            'cysparse.sparse.ll_mat_matrices',
            'cysparse.sparse.csr_mat_matrices',
            'cysparse.sparse.csc_mat_matrices',
            'cysparse.sparse.ll_mat_views',
            'cysparse.utils',
            #'cysparse.sparse.IO'
            'tests'
            ]

#packages_list=find_packages()

ext_modules = base_ext + sparse_ext

########################################################################################################################
# PACKAGE PREPARATION FOR EXCLUSIVE C EXTENSIONS
########################################################################################################################
# We only use the C files **without** Cython. In fact, Cython doesn't need to be installed.
if not use_cython:
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
    #ext_modules = cythonize(ext_modules), <- doesn't work with our settings... (combinations of .pxi and .pxd files)
    'ext_modules' : ext_modules,
    'package_dir' : {"cysparse": "cysparse"},
    'packages' : packages_list,
    'zip_safe' : False

}

if use_cython:
    setup_args['cmdclass'] = {'build_ext': build_ext}

setup(**setup_args)