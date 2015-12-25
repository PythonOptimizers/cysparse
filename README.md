# CySparse

Fast sparse matrix library for Python/Cython.

:warning: `COMPLEX256_T` is **no** longer supported :warning:

:cyclone: :cyclone: :cyclone: THIS VERSION IS BEING COMPLETELY REWRITTEN: :weary: YOU :weary: SHALL :no_entry: NOT :no_entry: USE IT! :cyclone: :cyclone: :cyclone:


I'm working on branch `feature/decoupling` which contains the :new: version of `CySparse`.
 
Nikolaj

I started to deal with sorted col/row indices for LL/CSC/CSR sparse matrices.

## Announcements

1. Unless there are some urgent needs (add the tag `priority` to the issues you want me to close urgently), I'll spend the next days (weeks) **decoupling** the code:

    - The template generation will be **completely** rewritten to ease the automation process (and avoid some recurrent human errors)
      and will be an autonomous project; :white_check_mark:
    - The code generation process will be revisited to allow the use of the Cython debugger 
      (it seems it is complicated to use the Python debugger on OSX). After one day trying... forget about cygdb. gbd itself works well and we now can add 
       debug symbols with a switch in `cysparse.cfg`. :white_check_mark:
    - The `linalg` part will be **completely** removed from `CySparse` and **each** interface with a solver will be an autonomous project (in their respective
      GitHub repositories). A common interface for all the solvers will be created and allow the interchange of solvers on the fly. :white_check_mark: (the linalg repositories 
      still have to be created though).
    - `CySparse` itself will have its API changed to better reflect the common use in the community (See `PySparse`, `NumPy` and `SciPy.sparse`).
    - A better mechanism will be implemented to allow the mix of special matrix cases (Symmetrical/general matrices, C-contiguous/non C-contiguous, etc). The aim
      is to introduce complex hermitian matrices (much later).

2. A centralized error/exception mechanism will be added.

3. Memory leaks will be taken seriously and hopefully I'll be able to avoid any memory leak (Valgrind anyone?). Some leaks were left on purpose: I wanted to test some
   Cython combinations before resolving these leaks (some ask for lots of work). These will be now taken care of.

4. There is a need for a better unit test generation: we need a meta meta generation. I have some ideas that I will test in the next weeks.

5. I have added **sorted** row/col indices for CSC/CSR matrices. This is a **BIG** change and needs some serious testing.
   You might experience some bugs because of this change... Let me know and I'll correct it. Thanks. This is still of actuality.

6. Documentation will be updated accordingly.

All this is very ambitious. I'll try to do it cleverly, step by step, knowing that I might get stuck somewhere before I can finish everything.
Each step will be done in dedicated branches so that the master and develop branches can be used as usual. Changes will only be committed in
major versions. So basically, there are two types of improvements going on: big non compatible changes (major versions) and business
as usual (minor versions).

## Dependencies

For the Python version:

- NumPy;

If you intend to generate the documention:

- Sphinx;
- sphinx_bootstrap_theme;

To run the tests:

- nose;

to run the performance tests:

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

- Version 0.3.0 released on Dec XX, 2015

  This is a major update.
  New documentation, split between users and developers. Pdf is now enabled.
  Completely new API.
  Thousands (7947) of unit tests added.
  
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

