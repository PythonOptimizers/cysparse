#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

#import numpy as np

ext_params = {}
ext_params['include_dirs'] = []  # [np.get_include()]
ext_params['extra_compile_args'] = ["-O2"]
ext_params['extra_link_args'] = []

sparse_ext_params = ext_params.copy()

ext_modules = [
  Extension("sparse_lib.sparse.ll_mat", ["sparse_lib/sparse/ll_mat.pyx"], **sparse_ext_params),
  Extension("sparse_lib.sparse.csr_mat", ["sparse_lib/sparse/csr_mat.pyx"], **sparse_ext_params),
]

setup(
    name='SparseLib',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    packages=['sparse_lib', 'sparse_lib.sparse']
)

