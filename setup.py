#!/usr/bin/env python

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
#                                                *** base ***
base_ext_params = ext_params.copy()
base_ext = [
    Extension(name="cysparse.cysparse_types",
              sources=["cysparse/cysparse_types.pxd", "cysparse/cysparse_types.pyx"]),
]

########################################################################################################################
#                                                *** sparse ***
sparse_ext_params = ext_params.copy()

sparse_ext = [
  Extension(name="cysparse.sparse.ll_mat",
            sources=["cysparse/sparse/ll_mat_details/ll_mat_multiplication.pxi",
                     "cysparse/sparse/ll_mat_details/ll_mat_assignment.pxi",
                     "cysparse/sparse/ll_mat_details/ll_mat_real_assignment_kernels.pxi",
                     "cysparse/sparse/ll_mat_details/ll_mat_real_multiplication_kernels.pxi",
                     "cysparse/sparse/ll_mat_details/ll_mat_transpose.pxi",
                     "cysparse/sparse/ll_mat.pxd",
                     "cysparse/sparse/ll_mat.pyx"], **sparse_ext_params),
  Extension(name="cysparse.sparse.sparse_mat",
            sources=["cysparse/sparse/sparse_mat.pxd", "cysparse/sparse/sparse_mat.pyx"], **sparse_ext_params),
  Extension(name="cysparse.sparse.csr_mat",
            sources=["cysparse/sparse/csr_mat.pxd", "cysparse/sparse/csr_mat.pyx"], **sparse_ext_params),
  Extension(name="cysparse.sparse.csc_mat",
            sources=["cysparse/sparse/csc_mat.pxd", "cysparse/sparse/csc_mat.pyx"], **sparse_ext_params),
  Extension(name="cysparse.sparse.ll_mat_view",
            sources=["cysparse.sparse.object_index.pxi",
                     "cysparse/sparse/ll_mat_view.pxd",
                     "cysparse/sparse/ll_mat_view.pyx"], **sparse_ext_params),
  Extension(name="cysparse.sparse.IO.mm",
            sources=["cysparse/sparse/IO/mm_read_file.pxi",
                     "cysparse/sparse/IO/mm_read_file2.pxi",
                     "cysparse/sparse/IO/mm_write_file.pxi",
                     "cysparse/sparse/IO/mm.pxd",
                     "cysparse/sparse/IO/mm.pyx"], **sparse_ext_params),
  #Extension("sparse.ll_vec", ["cysparse/sparse/ll_vec.pyx"], **sparse_ext_params)
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
ext_modules = base_ext + sparse_ext  + utils_ext + umfpack_ext


setup(name=  'SparseLib',
  #ext_package='cysparse', <- doesn't work with pxd files...
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  package_dir = {"cysparse": "cysparse"},
  packages=['cysparse',
            'cysparse.sparse',
            'cysparse.utils',
            'cysparse.solvers',
            'cysparse.solvers.suitesparse',
            'cysparse.sparse.IO'
            ]

)
