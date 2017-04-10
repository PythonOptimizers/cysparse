.. introduction:

====================================
Introduction
====================================

Welcome to :program:`CySparse`'s developers manual!

:program:`CySparse` is a fast sparse matrix library for :program:`Python`/:program:`Cython`.



Maintening a library as :program:`CySparse` is not a small task. This is partly due to:

- the mix of several programming languages (mainly :program:`Cython`, :program:`Python` and :program:`C`);
- the use of templated source files;
- the use of several external tools (`cygenja <https://github.com/PythonOptimizers/cygenja>`_, `Jinja2 <http://jinja.pocoo.org/>`_, `Sphinx <http://sphinx-doc.org/>`_, `LaTeX <https://www.latex-project.org/>`_, etc.);
- the optimized code (several code chuncks are highly optimized for speed);
- the coupling with `NumPy <http://www.numpy.org/>`_;
- the language `Cython <http://cython.org/>`_ that is not really mature yet. The team behind :program:`Cython` is doing a fantastic job but the language still has numerous bugs. If you've never used a compiler with bugs, welcome to this wonderful world were you might have
  to debug the compiler as much as your code (yep!);
