..  cysparse_intallation:

===================================
:program:`CySparse` installation
===================================

Installation
==============

Automatic
------------

Manual
--------


Different modes
==================

:program:`CySparse` can be compiled in different modes.

All compile parameters can be defined in the file :file:`CySparse/CySparse/cysparse_types.pxd`. Here are the lines that can be modified:

..  code-block:: text

    #################################################################################################
    #                                 *** COMPILATION CONSTANTS ***
    #################################################################################################
    # for huge matrices, we use unsigned long for size indices
    DEF USE_HUGE_MATRIX = 0
    # do we use double or float inside matrices?
    DEF USE_DOUBLE_PRECISION = 1

Parameters:
    
* ``USE_HUGE_MATRIX``: Setting it to 1 allows the use of ``long`` indices for the matrices, otherwise simple ``int`` indices are used [#internal_python_size_type]_.
<<<<<<< HEAD
* ``USE_DOUBLE_PRECISION``: Setting it to 1 triggers the use of ``double`` values insides the matrices, otherwise ``float`` values are stored.

Depending on your platform this might or migth not be relevant to you. We **strongly** suggest you to keep ``USE_DOUBLE_PRECISION`` to 1 and only change this parameter should you have memory problems
=======
* ``USE_DOUBLE_PRECISION``: Setting it to 1 triers the use of ``double`` values insides the matrices, otherwise ``float`` values are stored.

Depending on your platform this might or migth not be relevant to you. We **strongly** suest you to keep ``USE_DOUBLE_PRECISION`` to 1 and only change this parameter should you have memory problems
>>>>>>> master
because you use huge matrices. When using single precision, some parts of the library will not work. For instance, the ``SuiteSparse`` interface will not be enabled.


Depencies
============

Inconveniences
==============

**If** you transform the :program:`Cython` code yourself, sometimes :program:`Cython` can ask for a complete recompilation. Whenever this happens, it displays the following message:

..  code-block:: bash

    ValueError: XXX has the wrong size, try recompiling

where XXX is the first class that has the wrong size. The easiest way to deal with this is to recompile all the .pyx files again (you can force this by removing
all the .c files) [#cython_try_recompiling]_.

See Robert Bradshaw's `answer <https://groups.google.com/forum/?hl=en#!topic/cython-users/cOAVM0whJkY>`_. 
See also `enhancements distutils_preprocessing <https://github.com/cython/cython/wiki/enhancements-distutils_preprocessing>`_.


..  raw:: html

    <h4>Footnotes</h4>
    

..  [#internal_python_size_type] We use the ``Py_ssize_t`` type for indices when traversing Python objects (list, slices, ...).

..  [#cython_try_recompiling] The problem is interdependencies between source files that are not catched at compile time. Whenever :program:`Cython` can catch them at runtime, it throws this ``ValueError``.
