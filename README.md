# CySparse

Python/Cython library to replace PySparse.

:boom: MEGA BUG: we have a packaging/import problem. DO NO USE THIS VERSION :boom:

:white_check_mark: THIS VERSION SHOULD WORK BUT MORE TESTS ARE NEEDED! :white_check_mark:

I started to deal with sorted col/row indices for LL/CSC/CSR sparse matrices.

## Announcements

1. I have added **sorted** row/col indices for CSC/CSR matrices. This is a **BIG** change and needs some serious testing.
   You might experience some bugs because of this change... Let me know and I'll correct it. Thanks.

## Want to follow the implementation of CySparse?

See [Wiki](https://github.com/Funartech/cysparse/wiki) for details!

## Release history

- Version 0.1.5 released on July 18, 2015

  Added UMFPACK.

- Version 0.1.4 released on July 9, 2015

  Skipped immediately to version 0.1.4 (versions 0.1.2 and 0.1.3 were incorporated in version 0.1.0).

  Better support for `LLSparseMatrixView`s.

- Version 0.1.0 released on July 6, 2015

  First release with multiple types.

- Version 0.0.1 released on April 23, 2015

