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
cysparse_config.read('sparse_lib.cfg')

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
#                                                *** sparse ***
sparse_ext_params = ext_params.copy()

sparse_ext = [
  Extension("sparse_lib.sparse.ll_mat", ["sparse_lib/sparse/ll_mat.pxd", "sparse_lib/sparse/ll_mat.pyx"], **sparse_ext_params),
  Extension("sparse_lib.sparse.sparse_mat", ["sparse_lib/sparse/sparse_mat.pxd", "sparse_lib/sparse/sparse_mat.pyx"], **sparse_ext_params),
  Extension("sparse_lib.sparse.csr_mat", ["sparse_lib/sparse/csr_mat.pxd", "sparse_lib/sparse/csr_mat.pyx"], **sparse_ext_params),
  Extension("sparse_lib.sparse.csc_mat", ["sparse_lib/sparse/csc_mat.pxd", "sparse_lib/sparse/csc_mat.pyx"], **sparse_ext_params),
  Extension("sparse_lib.sparse.ll_mat_view", ["sparse_lib.sparse.object_index.pxi","sparse_lib/sparse/ll_mat_view.pxd", "sparse_lib/sparse/ll_mat_view.pyx"], **sparse_ext_params),
  Extension("sparse_lib.sparse.IO.mm", ["sparse_lib/sparse/IO/mm_read_file.pxi", "sparse_lib/sparse/IO/mm.pxd", "sparse_lib/sparse/IO/mm.pyx"], **sparse_ext_params),
  #Extension("sparse.ll_vec", ["sparse_lib/sparse/ll_vec.pyx"], **sparse_ext_params)
]

########################################################################################################################
#                                                *** utils ***
utils_ext = [
    Extension("sparse_lib.utils.equality", ["sparse_lib/utils/equality.pxd", "sparse_lib/utils/equality.pyx"], **sparse_ext_params),
]

########################################################################################################################
#                                                *** umfpack ***
umfpack_ext_params = ext_params.copy()
umfpack_ext_params['include_dirs'].extend(suitesparse_include_dirs)
#umfpack_ext_params['include_dirs'] = suitesparse_include_dirs
umfpack_ext_params['library_dirs'] = suitesparse_library_dirs
umfpack_ext_params['libraries'] = ['umfpack', 'amd']

umfpack_ext = [
    Extension("sparse_lib.solvers.suitesparse.umfpack",
              ['sparse_lib/solvers/suitesparse/umfpack.pxd',
               'sparse_lib/solvers/suitesparse/umfpack.pyx'],
              **umfpack_ext_params)
    ]


########################################################################################################################
# SETUP
########################################################################################################################
ext_modules = sparse_ext  + utils_ext + umfpack_ext


setup(name=  'SparseLib',
  #ext_package='sparse_lib', <- doesn't work with pxd files...
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  package_dir = {"sparse_lib": "sparse_lib"},
  packages=['sparse_lib',
            'sparse_lib.sparse',
            'sparse_lib.utils',
            'sparse_lib.solvers',
            'sparse_lib.solvers.suitesparse',
            'sparse_lib.sparse.IO'
            ]

)
