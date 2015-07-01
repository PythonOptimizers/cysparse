# CySparse

Python/Cython library to replace PySparse.

This is the multi-types version (last single type version: ce86f8476166f63d0af72e38a477a761aecf4a7c).

## Announcements

1. I'm in the process of replacing Cython properties by Python properties. This will take some time to settle down...
   You might experience some (stupid) bugs or slower code.

   Note to myself: **never** **ever** use automatic refactoring tools again... (they probably work great with a given code but **not** with
   my own source format...).

2. Cython memoryviews don't seem to be faster in our case. This should be assessed in the optimization phase. For the moment, I'll continue
   to code with NumPy's C-API. This is **not** the way Cython is intended by my tests indicate that this produces the fastest code. To be continued.

   Sorry for the fuzz.

   Stable version in sight: 15th of July.

## Want to follow the implementation of CySparse?

See [Wiki](https://github.com/Funartech/cysparse/wiki) for details!

## Release history

- Version 0.0.1 released on April 23, 2015

