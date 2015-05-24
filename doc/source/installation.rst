..  cysparse_intallation:

===================================
:program:`CySparse` installation
===================================

Installation
==============

Automatic
------------

Manual
--------



Depencies
============

:program:`CySparse`
---------------------

Documentation
-----------------

Unit testing
------------


Performance testing
----------------------

Inconveniences
==============

**If** you transform the :program:`Cython` code yourself, sometimes :program:`Cython` can ask for a complete recompilation. Whenever this happens, it displays the following message when trying to import the library 
into :program:`Python`:

..  code-block:: bash

    ValueError: XXX has the wrong size, try recompiling

where XXX is the first class that has the wrong size. The easiest way to deal with this is to recompile all the .pyx files again (you can force this by removing
all the .c files) [#cython_try_recompiling]_.

See Robert Bradshaw's `answer <https://groups.google.com/forum/?hl=en#!topic/cython-users/cOAVM0whJkY>`_. 
See also `enhancements distutils_preprocessing <https://github.com/cython/cython/wiki/enhancements-distutils_preprocessing>`_.

**If** you modify the templated code, some dependencies might be missing (and this is a **bug** you can report) in the (generated) ``setup.py`` file and require manual intervention, i.e. recompilation. The easiest way to go is to recompile everything from scratch. First delete the generated files:

..  code-block:: bash

    python generate_code.py -ac
    
where ``-ac`` stands for ``a``\ll and ``c``\lean. This will delete **all** generated ``.pxi``, ``.pxd`` and ``.pyx`` :program:`Cython` files. Then delete the generated :program:`C` files:

..  code-block:: bash

    python clean.py
    
This will delete **all** :program:`C` ``.c`` files. You can then recompile the library from scratch.



..  raw:: html

    <h4>Footnote</h4>
    

..  [#cython_try_recompiling] The problem is interdependencies between source files that are not catched at compile time. Whenever :program:`Cython` can catch them at runtime, it throws this ``ValueError``.
