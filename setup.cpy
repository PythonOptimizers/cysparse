#!/usr/bin/env python

# THIS FILE (setup.py) IS AUTOMATICALLY GENERATED
# Generate it with
# python generate_code -s

from setup.version import find_version, read
from setup.config import get_path_option

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

import ConfigParser
import os

####################################################################s####################################################
# INIT
########################################################################################################################
cysparse_config = ConfigParser.SafeConfigParser()
cysparse_config.read('cysparse.cfg')

numpy_include = np.get_include()

# SUITESPARSE
# Do we use it or not?
use_suitesparse = cysparse_config.getboolean('SUITESPARSE', 'use_suitesparse')
# find user defined directories
if use_suitesparse:
    suitesparse_include_dirs = get_path_option(cysparse_config, 'SUITESPARSE', 'include_dirs')
    suitesparse_library_dirs = get_path_option(cysparse_config, 'SUITESPARSE', 'library_dirs')

# MUMPS
# Do we use it or not?
use_mumps = cysparse_config.getboolean('MUMPS', 'use_mumps')
mumps_compiled_in_64bits = cysparse_config.getboolean('MUMPS', 'mumps_compiled_in_64bits')

# find user defined directories
if use_mumps:
    mumps_include_dirs = get_path_option(cysparse_config, 'MUMPS', 'include_dirs')
    mumps_library_dirs = get_path_option(cysparse_config, 'MUMPS', 'library_dirs')


########################################################################################################################
# EXTENSIONS
########################################################################################################################
include_dirs = [numpy_include, '.']

ext_params = {}
ext_params['include_dirs'] = include_dirs
# -Wno-unused-function is potentially dangerous... use with care!
ext_params['extra_compile_args'] = ["-O2", '-std=c99', '-Wno-unused-function']
ext_params['extra_link_args'] = []


########################################################################################################################
#                                                *** types ***
base_ext_params = ext_params.copy()
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
sparse_ext_params = ext_params.copy()

sparse_ext = [
  ######################
  # ### Sparse ###
  ######################
{% for index_type in index_list %}
  Extension(name="cysparse.sparse.sparse_utils.generic.generate_indices_@index_type@",
            sources=["cysparse/sparse/sparse_utils/generic/generate_indices_@index_type@.pxd",
                     "cysparse/sparse/sparse_utils/generic/generate_indices_@index_type@.pyx"],
            **sparse_ext_params),
  Extension(name="cysparse.sparse.sparse_utils.generic.sort_indices_@index_type@",
            sources=["cysparse/sparse/sparse_utils/generic/sort_indices_@index_type@.pxd",
                     "cysparse/sparse/sparse_utils/generic/sort_indices_@index_type@.pyx"],
            **sparse_ext_params),
{% endfor %}

{% for element_type in type_list %}
  Extension(name="cysparse.sparse.sparse_utils.generic.print_@element_type@",
            sources=["cysparse/sparse/sparse_utils/generic/print_@element_type@.pxd",
                     "cysparse/sparse/sparse_utils/generic/print_@element_type@.pyx"],
            **sparse_ext_params),
{% endfor %}

{% for index_type in index_list %}
    {% for element_type in type_list %}
  Extension(name="cysparse.sparse.sparse_utils.generic.find_@index_type@_@element_type@",
            sources=["cysparse/sparse/sparse_utils/generic/find_@index_type@_@element_type@.pxd",
                     "cysparse/sparse/sparse_utils/generic/find_@index_type@_@element_type@.pyx"],
            **sparse_ext_params),

  Extension(name="cysparse.sparse.sparse_utils.generic.matrix_translations_@index_type@_@element_type@",
            sources=["cysparse/sparse/sparse_utils/generic/matrix_translations_@index_type@_@element_type@.pxd",
                     "cysparse/sparse/sparse_utils/generic/matrix_translations_@index_type@_@element_type@.pyx"],
            **sparse_ext_params),
    {% endfor %}
{% endfor %}



  ######################
  # ### SparseMatrix ###
  ######################
  Extension(name="cysparse.sparse.s_mat",
            sources=["cysparse/sparse/s_mat.pxd",
                     "cysparse/sparse/s_mat.pyx"],
            **sparse_ext_params),

{% for index_type in index_list %}
    {% for element_type in type_list %}
  Extension(name="cysparse.sparse.s_mat_matrices.s_mat_@index_type@_@element_type@",
            sources=["cysparse/sparse/s_mat_matrices/s_mat_@index_type@_@element_type@.pxd",
                     "cysparse/sparse/s_mat_matrices/s_mat_@index_type@_@element_type@.pyx"],
            **sparse_ext_params),
    {% endfor %}
{% endfor %}

  ######################
  # ### LLSparseMatrix ###
  ######################
  Extension(name="cysparse.sparse.ll_mat",
            sources=["cysparse/sparse/ll_mat.pxd",
                     "cysparse/sparse/ll_mat.pyx"],
            **sparse_ext_params),

# TODO: add the possibility to **not** use tabu combinations...
{% for index_type in index_list %}
  {% for element_type in type_list %}
  Extension(name="cysparse.sparse.ll_mat_matrices.ll_mat_@index_type@_@element_type@",
            sources=["cysparse/sparse/ll_mat_matrices/ll_mat_@index_type@_@element_type@.pxd",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_@index_type@_@element_type@.pyx",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_addition_@index_type@_@element_type@.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_@index_type@_@element_type@.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_@index_type@_@element_type@.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_@index_type@_@element_type@.pxi",
  {% if index in mm_index_list and type in mm_type_list %}
                     "cysparse/sparse/ll_mat_matrices/ll_mat_IO/ll_mat_mm_@index_type@_@element_type@.pxi",
  {% endif %}
                     ],
            **sparse_ext_params),
  {% endfor %}
{% endfor %}

  ######################
  # ### CSRSparseMatrix ###
  ######################
{% for index_type in index_list %}
  {% for element_type in type_list %}
  Extension(name="cysparse.sparse.csr_mat_matrices.csr_mat_@index_type@_@element_type@",
            sources=["cysparse/sparse/csr_mat_matrices/csr_mat_@index_type@_@element_type@.pxd",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_@index_type@_@element_type@.pyx",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_helpers/csr_mat_multiplication_@index_type@_@element_type@.pxi",
                     "cysparse/sparse/csr_mat_matrices/csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_@index_type@_@element_type@.pxi",
                     ],
            **sparse_ext_params),
  {% endfor %}
{% endfor %}

  ######################
  # ### CSCSparseMatrix ###
  ######################
{% for index_type in index_list %}
  {% for element_type in type_list %}
  Extension(name="cysparse.sparse.csc_mat_matrices.csc_mat_@index_type@_@element_type@",
            sources=["cysparse/sparse/csc_mat_matrices/csc_mat_@index_type@_@element_type@.pxd",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_@index_type@_@element_type@.pyx",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_helpers/csc_mat_multiplication_@index_type@_@element_type@.pxi",
                     "cysparse/sparse/csc_mat_matrices/csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_@index_type@_@element_type@.pxi",
                     ],
            **sparse_ext_params),
  {% endfor %}
{% endfor %}

  ######################
  # ### LLSparseMatrixView ###
  ######################

{% for index_type in index_list %}
  {% for element_type in type_list %}
  Extension(name="cysparse.sparse.ll_mat_views.ll_mat_view_@index_type@_@element_type@",
            sources=["cysparse/sparse/ll_mat_views/ll_mat_view_@index_type@_@element_type@.pxd",
                     "cysparse/sparse/ll_mat_views/ll_mat_view_@index_type@_@element_type@.pyx"],
            **sparse_ext_params),
  {% endfor %}
{% endfor %}

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
{% for index_type in index_list %}
  {% for element_type in complex_list %}
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.h_mat_@index_type@_@element_type@",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/h_mat_@index_type@_@element_type@.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/h_mat_@index_type@_@element_type@.pyx"],
            **sparse_ext_params),
  {% endfor %}
{% endfor %}

  ######################
  # ### ConjugatedSparseMatrix ###
  ######################
{% for index_type in index_list %}
  {% for element_type in complex_list %}
  Extension(name="cysparse.sparse.sparse_proxies.complex_generic.conj_mat_@index_type@_@element_type@",
            sources=["cysparse/sparse/sparse_proxies/complex_generic/conj_mat_@index_type@_@element_type@.pxd",
                     "cysparse/sparse/sparse_proxies/complex_generic/conj_mat_@index_type@_@element_type@.pyx"],
            **sparse_ext_params),
  {% endfor %}
{% endfor %}
]
########################################################################################################################
#                                                *** utils ***
utils_ext = [
    Extension("cysparse.utils.equality", ["cysparse/utils/equality.pxd", "cysparse/utils/equality.pyx"], **sparse_ext_params),
]

########################################################################################################################
#                                                *** SuiteSparse ***
if use_suitesparse:
    umfpack_ext_params = ext_params.copy()
    umfpack_ext_params['include_dirs'].extend(suitesparse_include_dirs)
    #umfpack_ext_params['include_dirs'] = suitesparse_include_dirs
    umfpack_ext_params['library_dirs'] = suitesparse_library_dirs
    umfpack_ext_params['libraries'] = ['umfpack', 'amd']

    umfpack_ext = [
{% for index_type in umfpack_index_list %}
  {% for element_type in umfpack_type_list %}
        Extension(name="cysparse.linalg.suitesparse.umfpack.umfpack_@index_type@_@element_type@",
                  sources=['cysparse/linalg/suitesparse/umfpack/umfpack_@index_type@_@element_type@.pxd',
                           'cysparse/linalg/suitesparse/umfpack/umfpack_@index_type@_@element_type@.pyx'], **umfpack_ext_params),
    {% endfor %}
{% endfor %}
        ]

#../lib/libsmumps.so ../lib/libmumps_common.so  -L../PORD/lib/ -lpord  -L../libseq -lmpiseq -lblas -lpthread


if use_mumps:
    mumps_ext = []
{% for index_type in mumps_index_list %}
  {% for element_type in mumps_type_list %}
    mumps_ext_params_@index_type@_@element_type@ = ext_params.copy()
    mumps_ext_params_@index_type@_@element_type@['include_dirs'].extend(mumps_include_dirs)
    mumps_ext_params_@index_type@_@element_type@['library_dirs'] = mumps_library_dirs
    mumps_ext_params_@index_type@_@element_type@['libraries'] = [] # 'scalapack', 'pord']

    mumps_ext_params_@index_type@_@element_type@['libraries'].append('@element_type|cysparse_real_type_to_mumps_family@mumps')
    mumps_ext_params_@index_type@_@element_type@['libraries'].append('mumps_common')
    mumps_ext_params_@index_type@_@element_type@['libraries'].append('pord')
    mumps_ext_params_@index_type@_@element_type@['libraries'].append('mpiseq')
    mumps_ext_params_@index_type@_@element_type@['libraries'].append('blas')
    mumps_ext_params_@index_type@_@element_type@['libraries'].append('pthread')

    mumps_ext.append(

        Extension(name="cysparse.linalg.mumps.mumps_@index_type@_@element_type@",
                  sources=['cysparse/linalg/mumps/mumps_@index_type@_@element_type@.pxd',
                           'cysparse/linalg/mumps/mumps_@index_type@_@element_type@.pyx'], **mumps_ext_params_@index_type@_@element_type@))
  {% endfor %}
{% endfor %}


########################################################################################################################
# SETUP
########################################################################################################################
packages_list = ['cysparse',
            'cysparse.types',
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
            'cysparse.linalg.mumps',
            #'cysparse.sparse.IO'
            ]

ext_modules = base_ext + sparse_ext

if use_suitesparse:
    # add suitsparse package
    ext_modules += umfpack_ext
    packages_list.append('cysparse.linalg.suitesparse')
    packages_list.append('cysparse.linalg.suitesparse.umfpack')

if use_mumps:
    # add mumps
    ext_modules += mumps_ext
    packages_list.append('cysparse.linalg.mumps')

setup(name=  'CySparse',
  version=find_version(os.path.realpath(__file__), 'cysparse', '__init__.py'),
  #ext_package='cysparse', <- doesn't work with pxd files...
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  package_dir = {"cysparse": "cysparse"},
  packages=packages_list

)

