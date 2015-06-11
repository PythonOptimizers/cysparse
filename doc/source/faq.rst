.. _faq:

==========================================================
Frequently asked questions
==========================================================

- :ref:`Is it true than I can loose some precision when using lists? <pitfalls_list_and_precision>`

Pitfalls
=========

..  _pitfalls_list_and_precision:

- Yes. When using a ``list``, the :program:`CPython` API forces us to cast every real value to :program:`C` ``double``. Whenever using a type with more than ``double`` precision, you *might* loose some accuracy.
  To avoid this, use :program:`NumPy` arrays.


