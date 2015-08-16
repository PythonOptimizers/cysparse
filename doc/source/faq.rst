.. _faq:

==========================================================
Frequently asked questions
==========================================================

Questions are divided between:

- :ref:`faq_matrix_creation` : questions about matrix creations;
- :ref:`faq_optimization` : questions about the optimized (best) use of the library;

Questions:
==========

.. _faq_matrix_creation:

Matrix creation
---------------



..	_faq_optimization:

Optimization
-------------


Pitfalls
-----------

1. :ref:`Is it true than I can loose some precision when using lists? <pitfalls_list_and_precision>`

Answers:
=========

Matrix creation
----------------



Optimization
-------------

Pitfalls
----------

..  _pitfalls_list_and_precision:

1. Yes. When using a ``list``, the :program:`CPython` API forces us to cast every real value to :program:`C` ``double``. Whenever using a type with more than ``double`` precision, you *might* loose some accuracy.
   To avoid this, use :program:`NumPy` arrays.


