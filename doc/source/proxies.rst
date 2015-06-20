..  _sparse_matrix_proxies:

=====================
Sparse Matrix Proxies
=====================

This section describes the sparse matrix proxies available in
:program:`Cysparse`. Sometimes, we can use a substitute for certain types of matrices instead of creating a full scale matrix. For instance, instead of creating the transpose of a matrix `A`, one can use `A.T`. `A.T` is a 
proxy and is thus **not** a real matrix. As such, it cannot replace a matrix in all circonstances but it can be used for the following operations:

- printing;
- accessing a value;
- multiplying with a :program:`NumPy` vector;
- accessing basic attributes of a matrix (`shape`, `ncol`, `nrow`, ...); 

Available proxies
==================

Three basic proxies are available:

- the transpose matrix proxy (:class:`TransposedSparseMatrix` given by the ``.T`` attribute);
- the conjugate transpose matrix proxy (:class:`ConjugateTransposedSparseMatrix` given by the ``.H`` attribute) and
- the conjugate matrix proxy (:class:`ConjugatedSparseMatrix` given by the ``.conj`` attribute).

Each of them is unique (i.e. the user is not supposed to copy them) and automagically updates whenever the original matrix is changed (for :class:`LLSparseMatrix` matrices).

These proxies are available for **all** sparse matrix and proxy types when they make sense. This means that the following code is legit:

..  code-block:: python

    A = NewSparseMatrix(...)
    
    b = np.array(...)
    
    A.T.H.conj.H.H * b
    
The expression ``A.T.H.conj.H.H`` has to be read for left to right: we start by taking the transpose matrix of matrix ``A``, then the conjugate transpose matrix of the transpose matrix of matrix ``A`` etc. The chaining of proxies can be as long as memory permits and no penalty occurs, i.e. each proxy is only instantiated once.

The ``H`` and ``conj`` proxies are **only** available for complex matrices, i.e. matrices with a complex ``dtype``.

..  warning::  The ``H`` and ``conj`` proxies are **only** available for complex matrices.
    

Basic operations
=================


What if I need the full scale corresponding matrix?
====================================================

For all proxies, the method ``copy_matrix()`` is available:

..  code-block:: python

    A = NewSparseMatrix(...)
    
    Transpose_proxy = A.T
    
    # real matrix
    T = Transpose_proxy.copy_matrix()
    
``T`` is now a real matrix of the same type as the original ``A`` matrix.    
