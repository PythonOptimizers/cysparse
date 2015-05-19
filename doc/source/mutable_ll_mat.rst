.. _ll_mat:

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

