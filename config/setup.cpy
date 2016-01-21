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

########################################################################################################################
# EXTENSIONS
########################################################################################################################
include_dirs = [numpy_include, '.']

ext_params = {}
ext_params['include_dirs'] = include_dirs
# -Wno-unused-function is potentially dangerous... use with care!
# '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION': doesn't work with Cython... because it **does** use a deprecated version...


if not use_debug_symbols:
    ext_params['extra_compile_args'] = ["-O2", '-std=c99', '-Wno-unused-function']
    ext_params['extra_link_args'] = []
else:
    ext_params['extra_compile_args'] = ["-g", '-std=c99', '-Wno-unused-function']
    ext_params['extra_link_args'] = ["-g"]

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
                     #"cysparse/sparse/ll_mat_matrices/ll_mat_IO/ll_mat_mm_@index_type@_@element_type@.pxi",
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

  ######################
  # ### OpProxy ###
  ######################
  Extension(name="cysparse.sparse.operator_proxies.op_proxy",
            sources=["cysparse/sparse/operator_proxies/op_proxy.pxd",
                     "cysparse/sparse/operator_proxies/op_proxy.pyx"],
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
{% raw %}
    'author_email' : 'cysparse\@TODO.com',
{% endraw %}
    'maintainer' : "CySparse Developers",
{% raw %}
    'maintainer_email' : "dominique.orban@gerad.ca",
{% endraw %}
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
