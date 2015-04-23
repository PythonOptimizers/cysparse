..  sparse_matrix_formats:

=====================
Sparse Matrix Formats
=====================

This section describes the sparse matrix storage schemes available in
:program:`Cysparse`. In the next sections, we cover sparse matrix creation, population, view and conversion.

Basically, we use one general sparse matrix format to *create* a general sparse matrix: the :class:`LLSparseMatrix` class. This class has
a rich API and allows multiple ways to create and transform a sparse matrix. Once the matrix is ready to be used, it can be appropriately converted 
into a specialized sparse matrix format that is optimized for the needed operations. The :class:`LLSparseMatrix` is also the only type of matrices that
can be modified while the other types of matrices are **immutable** for efficiency.

Here is a list of existing formats and their basic use:

- Linked-list format (LL): a convenient format for creating and populating
  a sparse matrix, whether symmetric or general.
- Compressed sparse row format (CSR): a format designed to speed up
  matrix-vector products.
- Compressed sparse column format (CSC): a format designed to speed up
  vector-matrix products. 

The CSR and CSC formats are complementary and can be viewed as respectively a row- and column view of a sparse matrix.

These formats are well known in the community and you can find scores of documents about them on the internet.

The LL sparse format  in details
=======================================

This format is implemented by the :class:`LLSparseMatrix` class.


The CSR sparse format in details
=========================================

This format is implemented by the :class:`CSRSparseMatrix` class.


The CSC sparse format in details
========================================

This format is implemented by the :class:`CSCSparseMatrix` class.

The CSC sparse matrix format is exactly the same as the CSR sparse matrix format but column-wise. Given a matrix :math:`A` and a CSR representation of this matrix is exactly the same as a CSC respresentation 
of the transposed matrix :math:`A^t`, i.e.

..  math::
    \textrm{CSR}(A) = \textrm{CSC}(A^t)

and everything we wrote about the CSR format transposes to the CSC format by exchanging rows for columns and vice-versa.


