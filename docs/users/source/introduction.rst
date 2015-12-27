.. introduction_to_cy_sparse:

====================================
Introduction
====================================

Welcome to :program:`CySparse`'s users manual!

We tried to keep this manual as simple and short as possible. You can find much more detailed information in the developer's manual.
To see :program:`CySparse` in action, you can try the tutorials written in `IPython Notebooks <http://ipython.org/notebook.html>`_ or run the examples in the 
``examples`` directory.


What is :program:`CySparse`?
=============================

:program:`CySparse` is a fast sparse matrix library for :program:`Python`/:program:`Cython`.



:program:`PySparse` legacy
============================

:program:`CySparse` can be seen as the replacement of the :program:`PySparse` library with **lots** of improvements. Instead of improving the existing :program:`PySparse` library written in :program:`C` and :program:`CPython`, 
we decided to rewrite it from scratch in :program:`Cython`. This allows us a shorter developement cycle and to hook existing :program:`C` and :program:`C++` libraries very easily to :program:`CySparse` 
(see the `PythonOptimizers <https://github.com/PythonOptimizers>`_ project for a full list of compatible Solvers wrappers).

In short, :program:`CySparse` is :program:`PySparse` but on steroids!

:program:`PySparse` vs :program:`CySparse`
===========================================

Even if :program:`CySparse` is (strongly) inspired from :program:`PySparse`, there are notable differences. In short, :program:`CySparse`:

- allows the use of matrices with **different types** of indices and elements at **run time** (see ...);
- is **faster** than :program:`PySparse` (see our benchmarks);
- uses **matrix views** - a very light proxy object - that represent parts of a matrix **without** the need to copy any element (see...);
- uses **matrix proxies** - an even lighter proxy object - that represent some common transformation of a matrix (like the transposed of a matrix) **without** the need to copy any element (see  ...); 
- has more **syntactic sugar**, like ``A * b, b * A, A.T * b`` etc. 
- has a **symmetric** storage scheme for **all** its matrix types.
- doesn't use masks.
- has lots of unit tests.
- is well integrated with some of the best solvers (`SuiteSparse <http://faculty.cse.tamu.edu/davis/suitesparse.html>`_, `MUMPS <http://mumps.enseeiht.fr/>`_, `qr_mumps <http://buttari.perso.enseeiht.fr/qr_mumps/>`_, ...).
- is well documented.
- can be used in :program:`Python` **and** :program:`Cython`.
- has less dependencies.

Both libraries define similar but also different matrix classes: 

=========================================   ======================================================   ============================================
Matrix type                                 :program:`PySparse`                                      :program:`CySparse` 
=========================================   ======================================================   ============================================
Linked-List Format                          ``ll_mat``, ``ll_mat_sym``, ``PysparseMatrix``           ``LLSparseMatrix``
Compressed Sparse Row Format                ``csr_mat``                                              ``CSRSparseMatrix``
Compressed Sparse Column Format             -                                                        ``CSCSparseMatrix``
Sparse Skyline Format                       ``sss_mat``                                              -
Compressed Sparse Row and Column Format     -                                                        ``CSBSparseMatrix`` (later)
=========================================   ======================================================   ============================================
    
What is the maturity level of :program:`CySparse`?
==================================================

If you don't mix matrices with different types and use vector multiplication, :program:`CySparse` has been tested quite heavily but as we only started in 2016, we cannot say that lots of users already tried the library. 
If you stick to the basics, it probably will work.
The rest is work in progress. We are planning to add more tests soon and to test extensively its integration with several well known solvers.

..  warning:: Do not use this library for critical tasks!

The benchmarks also shows that the library is fast, much faster than its competitors.


License
========

The :program:`CySparse` library is released under the `GNU Lesser General Public License <http://www.gnu.org/licenses/lgpl-3.0.en.html>`_ (LGPL), version 3.
