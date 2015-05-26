#!/usr/bin/env python

# THIS FILE (setup.py) IS AUTOMATICALLY GENERATED
# Generate it with
# python generate_code -s

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

import ConfigParser


def get_path_option(config, section, option):
    """
    Get path(s) from an option in a section of a ``ConfigParser``.

    Args:
        config (ConfigParser): Configuration.
        section (str): Section ``[section]`` in the ``config`` configuration.
        option (str): Option ``option`` key.

    Returns:
        One or several paths ``str``\s in a ``list``.
    """
    import os
    try:
        val = config.get(section,option).split(os.pathsep)
    except:
        val = None
    return val

####################################################################s####################################################
# INIT
########################################################################################################################
cysparse_config = ConfigParser.SafeConfigParser()
cysparse_config.read('cysparse.cfg')

numpy_include = np.get_include()

# find user defined directories
suitesparse_include_dirs = get_path_option(cysparse_config, 'SUITESPARSE', 'include_dirs')
suitesparse_library_dirs = get_path_option(cysparse_config, 'SUITESPARSE', 'library_dirs')

########################################################################################################################
# EXTENSIONS
########################################################################################################################
include_dirs = [numpy_include, '.']

ext_params = {}
ext_params['include_dirs'] = include_dirs
ext_params['extra_compile_args'] = ["-O2"]
ext_params['extra_link_args'] = []

########################################################################################################################
#                                                *** types ***
base_ext_params = ext_params.copy()
base_ext = [
    Extension(name="cysparse.types.cysparse_types",
              sources=["cysparse/types/cysparse_types.pxd", "cysparse/types/cysparse_types.pyx"]),
    Extension(name="cysparse.types.cysparse_numpy_types",
              sources=["cysparse/types/cysparse_numpy_types.pxd", "cysparse/types/cysparse_numpy_types.pyx"]),
    ]

########################################################################################################################
#                                                *** sparse ***
sparse_ext_params = ext_params.copy()

sparse_ext = [
  #Extension(name="cysparse.sparse.ll_mat",
  #          sources=["cysparse/sparse/ll_mat_details/ll_mat_multiplication.pxi",
  #                   "cysparse/sparse/ll_mat_details/ll_mat_assignment.pxi",
  #                   "cysparse/sparse/ll_mat_details/ll_mat_real_assignment_kernels.pxi",
  #                   "cysparse/sparse/ll_mat_details/ll_mat_real_multiplication_kernels.pxi",
  #                   "cysparse/sparse/ll_mat_details/ll_mat_transpose.pxi",
  #                   "cysparse/sparse/ll_mat.pxd",
  #                   "cysparse/sparse/ll_mat.pyx"], **sparse_ext_params),
  #Extension(name="cysparse.sparse.sparse_mat",
  #          sources=["cysparse/sparse/sparse_mat.pxd", "cysparse/sparse/sparse_mat.pyx"], **sparse_ext_params),
  #Extension(name="cysparse.sparse.csr_mat",
  #          sources=["cysparse/sparse/csr_mat.pxd", "cysparse/sparse/csr_mat.pyx"], **sparse_ext_params),
  #Extension(name="cysparse.sparse.csc_mat",
  #          sources=["cysparse/sparse/csc_mat.pxd", "cysparse/sparse/csc_mat.pyx"], **sparse_ext_params),
  #Extension(name="cysparse.sparse.ll_mat_view",
  #          sources=["cysparse.sparse.object_index.pxi",
  #                   "cysparse/sparse/ll_mat_view.pxd",
  #                   "cysparse/sparse/ll_mat_view.pyx"], **sparse_ext_params),
  #Extension(name="cysparse.sparse.IO.mm",
  #          sources=["cysparse/sparse/IO/mm_read_file.pxi",
  #                   "cysparse/sparse/IO/mm_read_file2.pxi",
  #                   "cysparse/sparse/IO/mm_write_file.pxi",
  #                   "cysparse/sparse/IO/mm.pxd",
  #                   "cysparse/sparse/IO/mm.pyx"], **sparse_ext_params),
  #Extension("sparse.ll_vec", ["cysparse/sparse/ll_vec.pyx"], **sparse_ext_params)
]

########################################################################################################################
#                                                *** NEW sparse ***

new_sparse_ext = [
  ######################
  # ### Sparse ###
  ######################
{% for index_type in index_list %}
  Extension(name="cysparse.sparse.sparse_utils.generate_indices_@index_type@",
            sources=["cysparse/sparse/sparse_utils/generate_indices_@index_type@.pxd",
                     "cysparse/sparse/sparse_utils/generate_indices_@index_type@.pyx"],
            **sparse_ext_params),
{% endfor %}

{% for index_type in index_list %}
    {% for element_type in type_list %}
  Extension(name="cysparse.sparse.sparse_utils.find_@index_type@",
            sources=["cysparse/sparse/sparse_utils/find_@index_type@_@element_type@.pxd",
                     "cysparse/sparse/sparse_utils/find_@index_type@_@element_type@.pyx"],
            **sparse_ext_params),
    {% endfor %}
{% endfor %}



  #Extension(name="cysparse.sparse.ll_mat",
  #          sources=["cysparse/sparse/ll_mat_details/ll_mat_multiplication.pxi",
  #                   "cysparse/sparse/ll_mat_details/ll_mat_assignment.pxi",
  #                   "cysparse/sparse/ll_mat_details/ll_mat_real_assignment_kernels.pxi",
  #                   "cysparse/sparse/ll_mat_details/ll_mat_real_multiplication_kernels.pxi",
  #                   "cysparse/sparse/ll_mat_details/ll_mat_transpose.pxi",
  #                   "cysparse/sparse/ll_mat.pxd",
  #                   "cysparse/sparse/ll_mat.pyx"], **sparse_ext_params),
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
                     "cysparse/sparse/ll_mat_matrices/ll_mat_helpers/ll_mat_multiplication_@index_type@_@element_type@.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_assignment_kernel_@index_type@_@element_type@.pxi",
                     "cysparse/sparse/ll_mat_matrices/ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_@index_type@_@element_type@.pxi"
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
  Extension(name="cysparse.sparse.t_mat",
            sources=["cysparse/sparse/t_mat.pxd",
                     "cysparse/sparse/t_mat.pyx"],
            **sparse_ext_params),
]
########################################################################################################################
#                                                *** utils ***
utils_ext = [
    Extension("cysparse.utils.equality", ["cysparse/utils/equality.pxd", "cysparse/utils/equality.pyx"], **sparse_ext_params),
]

########################################################################################################################
#                                                *** umfpack ***
umfpack_ext_params = ext_params.copy()
umfpack_ext_params['include_dirs'].extend(suitesparse_include_dirs)
#umfpack_ext_params['include_dirs'] = suitesparse_include_dirs
umfpack_ext_params['library_dirs'] = suitesparse_library_dirs
umfpack_ext_params['libraries'] = ['umfpack', 'amd']

umfpack_ext = [
    Extension(name="cysparse.solvers.suitesparse.umfpack",
              sources=['cysparse/solvers/suitesparse/umfpack.pxd',
                       'cysparse/solvers/suitesparse/umfpack.pyx'], **umfpack_ext_params)
    ]


########################################################################################################################
# SETUP
########################################################################################################################
ext_modules = base_ext +  new_sparse_ext # + utils_ext + umfpack_ext


setup(name=  'SparseLib',
  #ext_package='cysparse', <- doesn't work with pxd files...
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  package_dir = {"cysparse": "cysparse"},
  packages=['cysparse',
            'cysparse.types',
            'cysparse.sparse',
            'cysparse.sparse.sparse_utils',
            'cysparse.sparse.s_mat_matrices',
            'cysparse.sparse.ll_mat_matrices',
            'cysparse.sparse.ll_mat_views',
            'cysparse.utils',
            #'cysparse.solvers',
            #'cysparse.solvers.suitesparse',
            #'cysparse.sparse.IO'
            ]

)
