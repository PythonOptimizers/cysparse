..  _sparse_matrix_proxies:

=====================
Sparse Matrix Proxies
=====================

This section describes the sparse matrix proxies available in
:program:`Cysparse`. Sometimes, we can use a substitute for certain types of matrices instead of creating a full scale matrix. For instance, instead of creating the transpose of a matrix `A`, one can use `A.T`. `A.T` is a 
proxy and is thus **not** a real matrix. As such, it cannot replace a matrix in all circonstances but it can be used for instance for the following operations:

- printing;
- accessing a value;
- multiplying with a :program:`NumPy` vector;
- accessing basic attributes of a matrix (`shape`, `ncol`, `nrow`, ...); 

Available proxies
==================

Three basic proxies [#proxy_is_called_sparse_matrix]_ are available:

- the transpose matrix proxy (:class:`TransposedSparseMatrix` given by the ``.T`` attribute);
- the conjugate transpose matrix proxy (:class:`ConjugateTransposedSparseMatrix` given by the ``.H`` attribute) and
- the conjugate matrix proxy (:class:`ConjugatedSparseMatrix` given by the ``.conj`` attribute).

Each of them is unique (i.e. the user is not supposed to copy them) and automagically updates whenever the original matrix is changed (for :class:`LLSparseMatrix` matrices).

These proxies are available for **all** sparse matrix types and can even be combined together. 

   

Proxies can be chained
==============================

Proxies can be chained without problem:

..  code-block:: python

    A = LLSparseMatrix(...)
    
    A.T.conj.H.conj.T.T.T
    
makes sense and ``A.T.conj.H.conj.T.T.T`` has to be read for left to right: we first take the transpose of the matrix, then the conjugate of the transposed matrix, etc. 
What is the type of ``A.T.conj.H.conj.T.T.T``?

..  code-block:: python

    type(A.T.conj.H.conj.T.T.T)
    
returns ``cysparse.sparse.sparse_proxies.t_mat.TransposedSparseMatrix``, i.e. we have that ``A.T.conj.H.conj.T.T.T`` is nothing else than ``A.T``.
You can chain as many proxies as you whish without paying any (real) penalty: proxies are singletons and therefor are only created once.


Proxies are **not** idempotent
==============================

Of course, a proxy applied to itself **cannot** be itself because ``A.T`` and ``A.T.T`` represent different matrices in general. But it get worse than that [#worse_than_that]_.

Remember that proxies behave like matrices but are not real matrices? What happens when you take the tranpose of a tranposed matrix?

..  code-block:: python

    A = LLSparseMatrix(...)
    
    B = A.T.T

``B`` is the orginal matrix again and this is really what happens in :program:`CySparse`. ``B`` **is** a real matrix **not** a proxy. Thus what you can do with ``A.T`` is not the same as what you can do with ``A.T.T``.

Throw in the real versus the complex cases and things get even worse. To help the user write generic code, we allow in :program:`CySparse` the use of ``.H`` and ``.conj`` for real matrices [#H_and_conj_in_the_real_case]_.
This means that for the real case:

- ``A.conj`` is in fact ``A`` and
- ``A.H`` is in fact ``A.T``.

Not need to worry thought.
All this proxy manipulation is quite natural and allow for a smooth writing, quite comparable to the real mathematical notation. There are **only two** rules to remember:

- Whenever a mathematical combination of operators returns the original matrix, :program:`CySparse` also returns the original matrix;
- For the real case: ``A.conj`` is ``A`` and ``A.H`` is ``A.T``.

This is so important that we need to frame these two rules.

.. topic:: Rules for proxy manipulation

    **Rule 1**: Whenever a mathematical combination of operators returns the original matrix, :program:`CySparse` also returns the original matrix;
    
    **Rule 2**: With a real matrix ``A``:
    
        - ``A.conj`` is the same as ``A`` (rule 1) and
        - ``A.H`` is ``A.T``.

Basic operations
=================

[TO BE WRITTEN]

What if I need the full scale corresponding matrix?
====================================================

For all proxies, the method ``matrix_copy()`` is available:

..  code-block:: python

    A = LLSparseMatrix(...)
    
    Transpose_proxy = A.T
    
    # real matrix
    T = Transpose_proxy.matrix_copy()
    
``T`` is now a real matrix of the same type as the original ``A`` matrix.    

..  raw:: html

    <h4>Footnote</h4>

..  [#proxy_is_called_sparse_matrix] Despite being *proxies* and **not** matrices, we still give them the name ``...SparseMatrix``. 

..  [#worse_than_that] Don't worry, you'll be able to relax by the end of this paragraph. You will even understand our implementation choices.

..  [#H_and_conj_in_the_real_case] This is quite standard among sparse matrices libraries.

