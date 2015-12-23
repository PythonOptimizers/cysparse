.. introduction_to_cy_sparse:

====================================
Introduction to :program:`CySparse`
====================================

What is :program:`CySparse`?
=============================

Content
========


:program:`PySparse` legacy
============================

:program:`PySparse` vs :program:`CySparse`
===========================================

Even if :program:`CySparse` is (strongly) inspired from :program:`PySparse`, there are notable differences. In short, :program:`CySparse`:

- allows the use of matrices with **different types** of indices and elements at run time (see ...);
- is **faster** than :program:`PySparse` (see ...);
- uses **matrix views** - a very light proxy object - that represent parts of a matrix without the need to copy elements (see...);
- has more **syntaxic sugar**, like ``A * b, b * A, A.T * b`` etc. 
- has a **symmetric** version of **all** its matrix types.
- **doesn't use masks**.

Both libraries define similar but also different matrix classes: 

  =========================================   ======================================================   ============================================
  Matrix type                                 :program:`PySparse`                                      :program:`CySparse` 
  =========================================   ======================================================   ============================================
  Linked-List Format                          ``ll_mat``, ``ll_mat_sym``, ``PysparseMatrix``           ``LLSparseMatrix``
  Compressed Sparse Row Format                ``csr_mat``                                              ``CSRSparseMatrix``
  Compressed Sparse Column Format             -                                                        ``CSCSparseMatrix``
  Sparse Skyline Format                       ``sss_mat``                                              -
  Compressed Sparse Row and Column Format     -                                                        ``CSBSparseMatrix``
  =========================================   ======================================================   ============================================
    



License
========

