
..  _multiplication_with_numpy_vector:

================================================
Multiplication with a :program:`NumPy` vector
================================================

One very common matrix operation is to multiply a sparse matrix with a dense :program:`NumPy` vector. In fact, this operation is so common that we have written very specialized and optimized code for it.
Because :program:`NumPy` vectors are dense, the multiplication of a sparse matrix by a :program:`NumPy` vector returns a new (dense) :program:`NumPy`.

All :program:`NumPy` vector multiplication operations work accross **all** sparse matrix types (``LLSparseMatrix``, ``CSRSparseMatrix`` and ``CSCSparseMatrix``), 
**all** proxy types (``TransposedSparseMatrix``, ``ConjugateTransposedSparseMatrix`` and ``ConjugatedSparseMatrix``) and this for **all** **real** and **complex** element types.
Views **don't** implement matrix multiplication.

As all multiplications are **higly** optimized, you really should use the corresponding method and not use a combination of them equivalent to the one you seek. For instance, 

..  math::

    A^H * x = conj(A^T * conj(x))

Because :math:`A^H * x` **is** implemented, it will be much faster than :math:`conj(A^T * conj(x))`.

Here is the list of the available multiplications:

- :math:`A * x`: ``A.matvec(x)``;
- :math:`A^T * x`: ``A.matvec_transp(x)``;
- :math:`A^H * x`: ``A.matvec_htransp(x)``;
- :math:`conj(A) * x`: ``A.matvec_conj(x)``;

These operations can also be optained throught the corresponding proxies (with the same efficiency):

- ``A.matvec(x)`` is the same as ``A * x``;
- ``A.matvec_transp(x)`` is the same as ``A.T * x``;
- ``A.matvec_htransp(x)`` is the same as ``A.H * x``;
- ``A.matvec_conj(x)`` is the same as ``A.conj * x``;

Optimization
============

The following has been optimized:

- each type is using the dedicated corresponding :program:`C` functions (for instance, the corresponding `conj` function from the standard lib);
- each operation is done at :program:`C` level;
- no factor is copied in any way: we work on the raw :program:`C` arrays;
- strided vectors have special dedicated code;
- we used the best loops to compute the multiplication for each type of matrix;

Helpers
========

in_arg and out_arg

For compapility reasons, we also provide the following global functions:

- ``matvec(A, x)``;
- ``matvec_transp(A, x)``;
- ``matvec_htransp(A, x)``;
- ``matvec_conj(A, x)``.

Matrix ``A`` must be a sparse matrix, aka its type must inherit from :class:`SparseMatrix`. All methods accept a :program:`NumPy` vector but the first two methods also accepts a sparse vector (i.e. ``x`` can also be one
dimensional sparse matrix).

``matvec()``
==============


Syntactic sugar
----------------

``matvec_transp``
=================

``matvec_htransp``
===================

Real matrices
---------------


``matvec_conj``
==================


Real matrices
--------------

What about sparse vectors?
===========================

:program:`CySparse` doesn't have a special sparse vector class. However, you can use a simple ``SparseMatrix`` object to represent your vector:

..  code-block:: Python

    v = LLSparseMatrix(nrow=4, ncol=1)
    v.put_triplet([0, 2], [0, 0], [1.0, 2.0])

    A = LinearFillLLSparseMatrix(nrow=3, ncol=4)

    print v
    print A

    C = A * v
    print C
    
returns the expected results:

..  only:: html

    ..  code-block:: bash

        LLSparseMatrix [INT64_t, FLOAT64_t] of size=(4, 1) with 2 non zero values <Storage scheme: General and without zeros>
         1.000000  
           ---     
         2.000000  
           ---     


        LLSparseMatrix [INT64_t, FLOAT64_t] of size=(3, 4) with 12 non zero values <Storage scheme: General and without zeros>
         1.000000   2.000000   3.000000   4.000000  
         5.000000   6.000000   7.000000   8.000000  
         9.000000  10.000000  11.000000  12.000000  


        LLSparseMatrix [INT64_t, FLOAT64_t] of size=(3, 1) with 3 non zero values <Storage scheme: General and without zeros>
         7.000000  
        19.000000  
        31.000000  

..  only:: latex

    ..  code-block:: bash

        LLSparseMatrix [INT64_t, FLOAT64_t] of size=(4, 1) with 2 non zero values 
        <Storage scheme: General and without zeros>
         1.000000  
           ---     
         2.000000  
           ---     


        LLSparseMatrix [INT64_t, FLOAT64_t] of size=(3, 4) with 12 non zero values 
        <Storage scheme: General and without zeros>
         1.000000   2.000000   3.000000   4.000000  
         5.000000   6.000000   7.000000   8.000000  
         9.000000  10.000000  11.000000  12.000000  


        LLSparseMatrix [INT64_t, FLOAT64_t] of size=(3, 1) with 3 non zero values 
        <Storage scheme: General and without zeros>
         7.000000  
        19.000000  
        31.000000  

Of course, the result **is** a sparse matrix. Contrary to :program:`NumPy` vectors, you need to give the right dimensions for the vector:

..  code-block:: python

    v = LLSparseMatrix(nrow=1, ncol=4)
    A = LinearFillLLSparseMatrix(nrow=3, ncol=4)
    
    A * v
    
will result in 

..  code-block:: bash

    IndexError: Matrix dimensions must agree ([3, 4] * [1, 4]) 
    
and you need to use **two** indices to access its elements:

..  code-block:: python

    v = LLSparseMatrix(nrow=4, ncol=1)
    for i in range(4):
        print v[i, 0]
        
           
