..  cysparse_intallation:

===================================
:program:`CySparse` installation
===================================

Different versions
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

Setting ``USE_HUGE_MATRIX`` to 1 allows the use of ``long`` indices for the matrices, otherwise simple ``int`` indices are used [#internal_python_size_type]_.
Setting ``USE_DOUBLE_PRECISION`` to 1 triggers the use of ``double`` values insides the matrices, otherwise ``float`` values are stored.

Depending on your platform this might or migth not be relevant to you. We **strongly** suggest you to keep ``USE_DOUBLE_PRECISION`` to 1 and only change this parameter should you have memory problems
because you use huge matrices.


Depencies
============

..  raw:: html

    <h4>Footnote</h4>
    

..  [#internal_python_size_type] We use the ``Py_ssize_t`` type for indices when traversing Python objects (list, slices, ...).
