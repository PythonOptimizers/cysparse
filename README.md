# CySparse

Python/Cython library to replace PySparse.

This is the multi-types version (last single type version: ce86f8476166f63d0af72e38a477a761aecf4a7c).

## Announcements

1. I'm in the process of replacing Cython properties by Python properties. This will take some time to settle down...
   You might experience some (stupid) bugs or slower code.

   Note to myself: **never** **ever** use automatic refactoring tools again... (they probably work great with a given code but **not** with
   my own source format...).

2. Cython memoryviews don't seem to be faster in our case. This should be assessed in the optimization phase. For the moment, I'll continue
   to code with NumPy's C-API. This is **not** the way Cython is intended but my tests indicate that this produces the fastest code. To be continued.

   Sorry for the fuzz.

   Stable version in sight: 15th of July.

3. If we don't use masks, 100% of PySparse's `ll_mat` has been implemented in CySparse.

## Want to follow the implementation of CySparse?

See [Wiki](https://github.com/Funartech/cysparse/wiki) for details!

## Release history

- Version 0.1.4 released on July 9, 2015

  Skipped immediately to version 0.1.4 (versions 0.1.2 and 0.1.3 were incorporated in version 0.1.0).

  Better support for `LLSparseMatrixView`s.

- Version 0.1.0 released on July 6, 2015

  First release with multiple types.

- Version 0.0.1 released on April 23, 2015

