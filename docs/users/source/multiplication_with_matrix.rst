..  _multiplication_with_matrix:

================================================
Multiplication with a matrix
================================================

Multiplication by another matrix isn't as important as the multiplication of a sparse matrix with a vector. Therefor, only two general methods are provided [#more_specialized_multiplication_methods_exist]_ [#matrix_multiplicaiton_work_in_progress]_:

- :math:`A * B`: ``A.matdot(B)``;
- :math:`A^T * B`: ``A.matdot_transp(B)``;

with ``A`` being a sparse matrix (``LLSparseMatrix``) or a matrix proxy () and ``B`` being another sparse matrix or :program:`NumPy` matrix.

These operations can also be optained throught the corresponding proxies (with the same efficiency):

- ``A.matdot(B)`` is the same as ``A * B``;
- ``A.matvec_transp(B)`` is the same as ``A.T * B``;

Views **don't** implement matrix multiplication.

Multiplication with a :program:`NumPy` matrix
------------------------------------------------

Multiplication with a :program:`NumPy` *ndarray* or *matrix* returns a dense :program:`NumPy` ``ndarray``.

..  code-block:: Python

    A = LinearFillLLSparseMatrix(nrow=2, ncol=2)
    B = np.mat('4 3; 2 1', dtype=np.float64) # NumPy matrix

    print A
    print B
    
    C = A * B
    
    print C
    print type(C)
    
displays:

..  only:: html

    ..  code-block:: bash

        LLSparseMatrix [INT64_t, FLOAT64_t] of size=(2, 2) with 4 non zero values <Storage scheme: General and without zeros>
         1.000000   2.000000  
         3.000000   4.000000  


        [[ 4.  3.]
         [ 2.  1.]]
        [[  8.   5.]
         [ 20.  13.]]
        <type 'numpy.ndarray'>

..  only:: latex

    ..  code-block:: bash

        LLSparseMatrix [INT64_t, FLOAT64_t] of size=(2, 2) with 4 non zero values 
        <Storage scheme: General and without zeros>
         1.000000   2.000000  
         3.000000   4.000000  


        [[ 4.  3.]
         [ 2.  1.]]
        [[  8.   5.]
         [ 20.  13.]]
        <type 'numpy.ndarray'>

As you can see, despite the :program:`NumPy` matrix being of type ``matrix``, we return a :program:`NumPy` ``ndarray`` [#only_numpy_ndarrays]_.

This matrix multiplication only works when you multiply a sparse matrix by a :program:`NumPy` matrix: multiplying a :program:`NumPy` matrix by a :program:`CySparse` sparse matrix is **not** allowed!
Thus the muliplication ``B * A`` with the matrices above raises an exception.

..  warning:: Multiplication of a :program:`CySparse` sparse matrix by a :program:`NumPy` matrix is **not** allowed!

..  only:: html

    ..  rubric:: Footnotes
    
..  [#only_numpy_ndarrays] This is general: :program:`CySparse` **only** returns :program:`NumPy` ``ndarrays``.

..  [#more_specialized_multiplication_methods_exist] Some more mutliplication methods exist. In particular, the :class:`LLSparseMatrix` class offers some other multiplication possibilities.

..  [#matrix_multiplicaiton_work_in_progress] Matrix by matrix multiplication is not as well supported as matrix by vector multiplication. This is work in progress.


