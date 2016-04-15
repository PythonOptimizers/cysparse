# CySparse

A fast sparse matrix library for Python/Cython.

[![Build Status](https://travis-ci.com/PythonOptimizers/cysparse.svg?token=ydgwcgKueSZx3k7qYsxd&branch=develop)](https://travis-ci.com/PythonOptimizers/cysparse)


**Only** Python 2.7 is supported for now. We plan to support Python 3.3 soon.
 
## Dependencies

For the Python version:

- NumPy;

If you intend to generate the documention:

- Sphinx;
- sphinx_bootstrap_theme;

To run the tests:

- nose or unittest;

To run the performance tests:

[TODO]

For the Cython version, include everything needed for Python and add:

- Cython;
- cygenja;

## Installation

### Python version

1. Clone repository (git clone https://github.com/PythonOptimizers/cysparse.git) or copy source code.
2. Install Python dependencies.
3. Copy `cysparse_template.cfg` to `cysparse.cfg` and adapt it to your needs.
4. Python setup.py install. 


### Cython version

[TODO]

## Run tests

Invoke:

```bash
python run_tests.py
```

Try the ``-h`` switch to see what options are available. Note that you need to install the library to run the tests.

## Run performance tests

[TODO]

## Want to follow the implementation of CySparse?

See [Wiki](https://github.com/Funartech/cysparse/wiki) for details!

## Release history

- Version 0.7.2 released on Feb 17, 2016

  Added some tests for the operator proxies. Only for linear combinations of matrix-like objects. Now at 13859 tests.

- Version 0.7.1 released on Feb 16, 2016
  
  Linear combinations of Matrices/matrix-like objects are now possible.
  
- Version 0.7.0 released on Jan 20, 2016

  Introduction of Travis.
  create_conjugate_transpose() -> create_adjoint()
  
- Version 0.6.0 released on Jan 19, 2016

  Changed matvec_htransp -> matvec_adj and matdot_htransp -> matdot_adj.
  
- Version 0.5.0 released on Jan 09, 2016

  Introduction of Operator Proxies (SumOp, MulOp).
    
- Version 0.4.0 released on Dec 30, 2015

  All PyXXX_Check() functions now in s_mat.
  
- Version 0.3.0 released on Dec 27, 2015

  This is a major update.
  New documentation, split between users and developers. Pdf is now enabled.
  Completely new API.
  Thousands (10499) of unit tests added.
  
- Version 0.2.2 released on Dec 14, 2015

  Clean up.
  
- Version 0.2.0 released on Dec 14, 2015

  Use of ``cygenja``, decoupling of ``linalg``.
  
- Version 0.1.5 released on July 18, 2015

  Added UMFPACK.

- Version 0.1.4 released on July 9, 2015

  Skipped immediately to version 0.1.4 (versions 0.1.2 and 0.1.3 were incorporated in version 0.1.0).

  Better support for `LLSparseMatrixView`s.

- Version 0.1.0 released on July 6, 2015

  First release with multiple types.

- Version 0.0.1 released on April 23, 2015

