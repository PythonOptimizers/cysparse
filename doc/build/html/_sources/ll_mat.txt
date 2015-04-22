.. ll_mat:

==========================================================
:class:`LLSparseMatrix` and :class:`LLSparseMatrixView`
==========================================================



The :class:`LLSparseMatrix` class
==================================

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
    
    - efficiency: no copy is made and the :class:`LLSparseMatrixView` is a light object. TO BE COMPLETED
    
    - really fancy indexing: if you need a new matrix constructed by **any** combinations of rows and columns, including 
      index repetitions. TO BE COMPLETED.

Views of views
--------------


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

