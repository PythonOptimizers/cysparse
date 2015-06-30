# CySparse

Python/Cython library to replace PySparse.

This is the multi-types version (last single type version: ce86f8476166f63d0af72e38a477a761aecf4a7c).

## Announcements

1. I'm in the process of replacing Cython properties by Python properties. This will take some time to settle down...
   You might experience some (stupid) bugs or slower code.

   Note to myself: **never** **ever** use automatic refactoring tools again... (they probably work great with a given code but **not** with
   my own source format...).

2. CySparse will **probably (?)** need to be **completely** rewritten: Cython memoryviews seem to be 20+% faster than the old NumPy C-API used in CySparse
   if used properly. No stable version of CySparse in sight before **end July 2015**. The transition will be done piece by piece and is **not** my priority
   for the moment. I'll do some benchmarking to assess this process.

   **Update**: Hm, maybe for I don't know what reason, this speed up doesn't work with CySparse. After some tests, memory views where still slower...
   see #126.

## Want to follow the implementation of CySparse?

See [Wiki](https://github.com/Funartech/cysparse/wiki) for details!

## Release history

- Version 0.0.1 released on April 23, 2015

