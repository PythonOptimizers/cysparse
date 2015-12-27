.. _mutable_ll_mat:

=================================================================================================
Mutable matrices: :class:`LLSparseMatrix` and :class:`LLSparseMatrixView`
=================================================================================================



The :class:`LLSparseMatrix` class
==================================

The *mutable* :class:`LLSparseMatrix` class is the base class to **construct** and **populate** a matrix. With it you can easily add or delete elements, rows, columns, assign sub-matrices, etc. Once your matrix is constructed, 
you create a new *optimized* and *immutable* matrix from it. You can choose between CSR, CSC and CSB. Each has its strength and weaknesses and we cover them in depth in their respective sections.

..  warning:: The ``LLSparseMatrix`` class is **not** optimized for matrix operations!

Creation
----------

Population
-----------



Accessing elements
-------------------

Elements can be accessed individually or by batch, i.e. several elements can be accessed at the same time and stored in a container.

Individual access
^^^^^^^^^^^^^^^^^^

You can use the common ``[]`` operator:

..  code-block:: python

    L = NewLLSparseMatrix(...)
    
    e = L[i, j]

where ``i`` and ``j`` are integer indices. At all time, bounds are checked and an ``IndexError`` is raised if an index is out of bound. :program:`Cython` users can access the elements **whithout** bound checking if desired.

Notice that if one of the argument you pass to the ``[]`` operator is **not** an integer, you'll get an ``LLSparseMatrixView`` (see :ref:`ll_sparse_matrix_view`).

..  warning:: If one of the argument you pass to the ``[]`` operator is **not** an integer, you'll get an ``LLSparseMatrixView``

Batch access
^^^^^^^^^^^^^

Basically, you can take a submatrix (and store its elements in a ``NumPy`` two dimensionnal array or another ``LLSparseMatrix`` or get a view to it, see :ref:`ll_sparse_matrix_view` ) or a list of elements


..  _ll_sparse_matrix_view:

The :class:`LLSparseMatrixView` class
=======================================

..  topic:: What is an :class:`LLSparseMatrixView` good for?

    There are basically two reasons to use a view instead of a matrix:
    
    - efficiency: no matrix_copy is made and the :class:`LLSparseMatrixView` is a light object. TO BE COMPLETED
    
    - really fancy indexing: if you need a new matrix constructed by **any** combinations of rows and columns, including 
      index repetitions. TO BE COMPLETED.

Views of views
--------------

It is possible to have views of... views as the following code illustrates:

..  code-block:: python 

    A = MakeLLSparseMatrix(...)
    A_view1 = A[..., ...]
    A_view2 = A_view1[..., ...]

The second :class:`LLSparseMatrixView` is **not** a view on a view but a direct view on the original matrix ``A``. The only difference between the two objects ``A_view1`` and ``A_view2`` is that 
the indices given in the ``[..., ...]`` in ``A_view1[..., ...]`` refer to indices of ``A_view1`` **not** the original matrix ``A``.

An example will clarify this:

..  code-block:: python

    pass

References to the base :class:`LLSparseMatrix`
----------------------------------------------

Whenever a :class:`LLSparseMatrixView` is created, the corresponding :class:`LLSparseMatrix` has its 
internal CPython reference count incremented such that even if the matrix object is (implicitely of explicitely) deleted, it still
exist in Python internal memory and can even be retrieved again through the view:

..  code-block:: python

    A = MakeLLSparseMatrix(...)
    A_view = A[..., ...]
    
    del A  
    
    A_view[..., ...] = ...  # still works!
    
    A = A_view.get_matrix() # A points again to the original matrix  

In the code above, the :class:`LLSparseMatrix` pointed by the variable ``A`` on the first line has never been 
deleted from memory. If you also delete **all** :class:`LLSparseMatrixView` objects refering to the :class:`LLSparseMatrix` object,
then it is effictively deleted by the garbage collector. 

..  code-block:: python

    A = MakeLLSparseMatrix(...)
    A_view = A[..., ...]
    
    del A
    del A_view
    
    # matrix A is lost... and will be deleted by the garbage collector 

