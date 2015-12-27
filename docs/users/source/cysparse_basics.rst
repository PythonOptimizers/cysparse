..  _cysparse_basics:

=========================================================
Basics
=========================================================

[TO BE REWRITTEN]

To add: shape

..  only:: html
    
    This page presents the basics of :program:`CySparse`. It covers a lot but you don't need to understand all the details. Instead, treat it as a gentle warm up for the rest of the manual.

..  only:: latex

    This chapter presents the basics of :program:`CySparse`. It covers a lot but you don't need to understand all the details. Instead, treat it as a gentle warm up for the rest of the manual.

Sparse Matrices and other matrix-like objects
==============================================

The main objects of the :program:`CySparse` library are *sparse matrices*: matrices that typically don't contain much elements that are different that ``0``. Say otherwise, sparce matrices contain lots of ``0``. 
To gain space and for efficiency reasons we don't store these zero elements explicitly [#store_zero_elements]_. Sparse matrices behave like usual matrices but behind the scene, their implementation use lots of trickery to use 
partial arrays that only store non zero elements. Several implementations exists with each their pros and cons. For instance, if you want to multiply a sparse matrix with a vector, you should use a ``CSR`` matrix, 
i.e. a  *Compressed Sparse Row* format matrix. Don't worry, in :program:`CySparse`, there are only **3** types of sparse matrices: 

- Linked-list format (**LL**): a convenient format for creating and populating a sparse matrix.
- Compressed sparse row format (**CSR**): a format designed to speed up matrix-vector products.
- Compressed sparse column format (**CSC**): a format designed to speed up vector-matrix products.

Linked-list (LL) format matrices are well suited to **create** and/or **modify** a matrix: you can easily add or delete elements for such matrices. Compressed sparse row (CSR) and Compressed sparse column (CSC) formats
are more suited for optimized vector multiplication and are difficult to modify. In :program:`CySparse` we even don't allow you to modify CSR or CSC matrices!

Matrices are the big thing in :program:`CySparse` but these objects are still heavy and sometimes you don't really need to create them. Enter the matrix *proxies* and matrix *views*. These object behave like matrices but 
are **not** matrices. A small example will clarify this. Imagine you have a matrix :math:`A` and you need to multiply its transposed with a vector. One way would be to create the transposed matrix and multiply it by this vector.
Actually, you **don't** need to create the transpose of a matrix for that! :program:`CySparse` let you play with a proxy transposed matrix, a fake matrix if you wish:

..  code-block:: python

    A = ...  # this is a sparse matrix
    v = ...  # this is a vector
    
    w = A.T * v    

To compute the vector ``w``, :program:`CySparse` doesn't need to actually create a transposed matrix. The ``T`` attribute is used for the transposed matrix of ``A``. No computations are done. When the time comes to do 
some computations, like multiplying this transposed ``A.T`` with a vector ``v``, :program:`CySparse` behind the scene computes this product directly from the matrix ``A`` and the vector ``v`` **without** creating the 
transposed matrix ``A.T``. ``A.T`` behaves like a matrix but is not a matrix. If you want to modify the transposed, which is **not** allowed, like this:

..  code-block:: python

    A.T[1,4] = 6
    
it throws an exception:

..  only:: html

    ..  code-block:: bash

        TypeError: 'cysparse.sparse.sparse_proxies.t_mat.TransposedSparseMatrix' object does not support item assignment

..  only:: latex

    ..  code-block:: bash

        TypeError: 'cysparse.sparse.sparse_proxies.t_mat.TransposedSparseMatrix' 
        object does not support item assignment

The type of the ``A.T`` object seems complicated and it is. The reason is again efficiency. However, except when dealing with such error messages, the types of the objects don't matter for the users as :program:`CySparse`
takes care of such intricacies. For instance, the product seen above of the transposed with a vector looks very much like it is mathematically written:

..  math::

    A^t * v

There is only one more type of objects that behaves like a matrix but isn't: views. We tried to assign a new value to an element of the ``A.T`` proxy matrix and this is not allowed. It can happen that sometimes, you need to have 
access to a sub-matrix and even change some elements of this sub-matrix. Enter the **views** that as their name implies allow to have a *view* over a matrix:

..  code-block:: python

    A = ... # this is a sparse matrix
    B = A[3:5, 2:7]
    
    
``B`` is a *view* and corresponds to the sub-matrix of ``A`` made by its rows 4 to 5 (:program:`Python` starts couting elements from ``0``) and its columns 3 to 6. You probably recognized the *slice* notation and it is exactly that.

..  code-block:: python

    C = A[:, ::2]
    
``C`` corresponds to the sub-matrix of ``A`` made by all its rows and every two columns. Actually, views can be much more complicated than that:

..  code-block:: python

    D = A[[1, 1, 1, 2 , 0], [3, 2 , 1, 0]]
    
Here we use two lists and ``D`` corresponds to a matrix that is no longer a sub-matrix of ``A``: it is constructed by taking 3 times the second row, then adding the third one and then the first one combined with the first four columns in reversed order.
We could also write:

..  code-block:: python

    D = A[[1, 1, 1, 2 , 0], 0:4:-1]

What about modifying an element of a view?

..  code-block:: python

    B[1, 4] = 42
 
is perfectly fine. That is, because ``B`` is a :math:`2 \times 5` matrix. We could have stored the value ``42`` in the view but that would complicate things a little bit too much and we would loose all efficiency. Instead, when
you modify a view, you modify the original matrix! By assigning ``42`` to element :math:`(1,4)` of ``B`` you are in fact assigning ``42`` to element :math:`(4, 6)` of the original matrix ``A``!

Views only exist for **LL** format matrices but because you read carefully, you already knew that [#why_only_one_type_of_view]_.

That's it. You have already seen **all** the main objects of the library! Of course, we need to talk a little bit more about the details and the common uses.

You can read more on:

- LL matrices and views: :ref:`mutable_ll_mat`;
- CSC and CSR matrices: :ref:`immutable_mat`;
- proxy matrices: :ref:`sparse_matrix_proxies`.

Storage schemes
===============

[TO BE REWRITTEN]

Common strorage attributes
----------------------------

For efficiency reasons, :program:`CySparse` can use different storage schemes/methods to store matrices. For instance, symmetric matrices can be stored with only half of their elements. 

``store_symmetric``
"""""""""""""""""""""""

Symmetric matrices can be stored by only storing the **lower** triangular part and the diagonal of a matrix. To create a symmetric matrix, add the arguement ``store_symmetric=True`` to the call of one of the factory methods.
The attribute ``store_symmetric`` returns if this storage method is used or not. Thus, if ``store_symmetric`` is ``True``, you know that you deal with a symmetric matrix **and** that roughly only half of its elements are stored. If 
``store_symmetric`` is ``False``, it simply means that this storage scheme is not used. The matrix itself migth be symmetric or not.

``store_zero``
"""""""""""""""""""

By default non specified (implicit) elements are zero (``0``, ``0.0`` or ``0+0j``). :program:`CySparse` allow the user to store explicitely zeros. To explicitely store zeros, declare ``store_zero=True`` as an argument
in any factory method:

..  code-block:: python

    A = LLSparseMatrix(store_zero=True, ...)
    
The matrix ``A`` will store any zero explicitely as will any matrix created from it. You can access the value of this attribute:

..  code-block:: python

    A.store_zero
    
returns ``True`` for our example. This attribute is read-only and cannot be changed. If you want to temporarily exclude zeros in some operations, you can use the ``NonZeros`` context manager:

..  code-block:: python

    with NonZeros(A):
        # use some method to add entries to A but disregard zeros entries
        ...

This context manager temporarily set the ``store_zero`` attribute to ``False`` before restoring its inital value.

By default, ``store_zero`` is set to ``False``.

``is_mutable``
""""""""""""""""""

``is_mutable`` returns if the matrix can be modified or not. Note that for the moment, **only** an :class:`LLSparseMatrix` matrix can be modified.

memory
""""""""

Just for fun, you can ask each matrix type (``LLSparseMatrix``, ``CSCSparseMatrix`` and ``CSRSparseMatrix``) how much memory they actually use (``memory_real()``) versus the memory a corresponding full dense matrix
would need (``memory_virtual()``):

..  code-block:: python

    A = ArrowheadLLSparseMatrix(nrow=50, ncol=800, itype=INT64_T, dtype=COMPLEX128_T)

    print A.memory_real()
    print A.memory_virtual()

which returns:

..  code-block::
    
    34448
    5120000

The results are in **bits**. ``ArrowheadLLSparseMatrix`` creates a sparse matrix. In this case, this sparse matrix has 898 non zero elements. The difference is huge and this is one of the reason we **do** use special
representations of sparse matrices instead of storing all the zeros.

Note that this memory is **only** the memory needed for the internal :program:`C`-arrays, **not** the total memory needed to store the whole matrix object. 

You can also ask what memory **one** element needs to be stored internally.
For our example:

..  code-block:: python

    print A.memory_element()

returns 

..  code-block:: python

    128
    
128 bits for a ``COMPLEX128_T`` type is to be expected.



..  _basics_types:

Types
=======

Each library has been written with a main goal in mind. We tried to optimize :program:`CySparse` for speed. Its code is therefor highly optimized and we use *typed* :program:`C` variables internally. All elements inside 
a matrix have the same well-defined types. Even indices are typed! For efficiency reasons, we often don't allow XXX


You will not be surprised that in :program:`CySparse`, we **strongly** discourage the mixing of types. Actually, most operations require to deal with objets that have **exactly**
the same types for their indices **and** their elements. As a rule of thumb, try not to mix objects with different types. We call this our *Golden Rule*.

..  important:: The Golden Rule:

    Never mix matrices with different types! 
    
If you follow this rule, you will not run into troubles in :program:`CySparse`.    
    
:program:`NumPy` and :program:`SciPy` let you mix matrices with different types. You can even declare your own type for the elements of a matrix! This flexibility has a cost as you can see in our benchmarks. :program:`CySparse`
is not as flexible but is some order of magnitude more efficient.

[TO BE REWRITTEN]

Common type attributes
------------------------

``dtype`` and ``itype``
""""""""""""""""""""""""""

Each matrix (matrix-like) object has an internal index *type* and stores *typed* elements. Both types (enums) can be retrieved.
``dtype`` returns the type of the elements of the matrix and ``itype`` returns its index type.
 
See section :ref:`availabe_types` about the available types.

What to import
==============

For :program:`Python users`, **only one** file needs to be imported:

- :file:`ll_mat`: to create an ``LLSparseMatrix``.

In short, you only need this:

..  code-block:: python

    from cysparse.sparse.ll_mat import *

:program:`Cython`'s users can import any :program:`Cython` file in any order.

Factory methods
===============

[TO BE REWRITTEN]

Common content attributes
-----------------------------


``nrow`` and ``ncol``
"""""""""""""""""""""""""

``nrow`` and ``ncol`` give respectively the number of rows and columns. You also can grab both at the same time with the ``shape`` attribute:

..  code-block:: python

    A = ...
    A.shape == A.nrow, A.ncol  # is True
    
You can use ``nrow`` and ``ncol`` as arguments to construct a new matrix. Whenever the number of rows is equal to the number of columns, i.e. when the matrix is square, you can
instead use the argument ``size=...`` in most factory methods.

``nnz``
""""""""""

The ``nnz`` attribute returns the number of "non zeros" stored in the matrix. Notice that ``0`` could be stored if ``store_zero`` is set to ``True`` and if so, it will be counted in the number of "non zero" elements.
Whenever the symmetric storage scheme is used (``store_symmetric`` is ``True``), ``nnz`` only returns the number of "non zero" elements stored in the lower triangular part and the diagonal of the matrix, i.e. ``nnz`` 
returns exactly how many elements are stored internally.

..  warning:: ``nnz`` returns **exactly** the number of elements stored internally.

When using views, this attribute is **costly** to retrieve as it is systematically recomputed each time and we don't make any assomption on the views (views can represent matrices with rows and columns in any order and duplicated 
rows and columns any number of times). The number returned is the number of "non zero" elements stored in the equivalent matrix using the **same** storage scheme than viewed matrix.
    




``is_symmetric``
"""""""""""""""""""""""

[TODO in the code!!!]

Returns if the matrix is symmetric or not. While matrices using the symmetric storage (``store_symmetric == True``) are symmetric by definition and ``is_symmetric`` returns immediatly ``True``, this attribute is costly to 
compute in general.



Common string attributes
----------------------------

Some attributes are stored as ``C`` struct internally and can thus not be accessed from :program:`Python`. We do however provide some strings for the most important ones.

``base_type_str`` and ``full_type_str``
""""""""""""""""""""""""""""""""""""""""""

Each matrix or matrix-like object has its own type and type name defined as strings. For instance:

..  code-block:: python

    A = NewLLSparseMatrix(size=10, dtype=COMPLEX64_T, itype=INT32_T)
    print A.base_type_str
    print A.full_type_str
    
returns

..  code-block:: bash

    LLSparseMatrix
    LLSparseMatrix [INT32_t, COMPLEX64_t]

The type ``LLSparseMatrix`` is common among ``LL`` sparse format matrices while the ``full_type_str`` gives the specific details of the index and element types.



..  only:: html

    ..  rubric:: footnotes

..  [#store_zero_elements] In some cases thought, it **is** worth keeping **some** zeros explicitly. This can be done in :program:`CySparse` through the use of the ``store_zero`` parameter. Read further.
  
..  [#why_only_one_type_of_view] Remember that the LL format matrices are the only ones you can modify in :program:`CySparse`!












