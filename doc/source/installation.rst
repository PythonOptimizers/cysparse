..  cysparse_intallation:

===================================
:program:`CySparse` installation
===================================

Installation
==============

The installation can be done in a few simple steps:

1. Clone the repository;
2. Install the dependencies;
3. Generate the source code:

   Some parts of the library source code have to be generated. We use a script:

   ..  code-block:: bash

       python generate_code.py -a
        
   The switch ``-a`` stands for ``--all`` and generates the entire library. If you need help, try the ``-h`` switch.
    
4. Compile and install the library:

   Use the traditionnal:

   ..  code-block:: bash

       python setup.py install






Depencies
============

All :program:`Python` dependencies are described in the :file:`requirements.txt` files. You can easily install them all with:

..  code-block:: bash

    pip install -r requirements.txt

or a similar command. Other dependencies need some manual installation. Read further.


:program:`CySparse`
---------------------

- :program:`Cython`
- :program:`Jinja2`
- argparse
- fortranformat
- :program:`SuiteSparse` (for the moment, it not possible to install :program:`CySparse` **without** :program:`SuiteSparse`)

Documentation
-----------------

- :program:`Sphinx`
- sphinx-bootstrap-theme

Unit testing
------------

- :program:`PySparse`

Performance testing
----------------------

- :program:`PySparse`
- benchmark.py (https://github.com/optimizers/benchmark.py)

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

**If** you modify the templated code, some dependencies might be missing in the (generated) ``setup.py`` file and require manual intervention, i.e. recompilation. The easiest way to go is to recompile everything from scratch [#missing_dependencies_generated_templates]_. First delete the generated files:

..  code-block:: bash

    python generate_code.py -ac
    
where ``-ac`` stands for ``a``\ll and ``c``\lean. This will delete **all** generated ``.pxi``, ``.pxd`` and ``.pyx`` :program:`Cython` files. Then delete the generated :program:`C` files:

..  code-block:: bash

    python clean.py
    
This will delete **all** :program:`C` ``.c`` files. You can then recompile the library from scratch.



..  raw:: html

    <h4>Footnotes</h4>
    

..  [#cython_try_recompiling] The problem is interdependencies between source files that are not catched at compile time. Whenever :program:`Cython` can catch them at runtime, it throws this ``ValueError``.

..  [#missing_dependencies_generated_templates] Interdependencies between generated templates are **not** monitored. Instead of recompiling everything from scratch, you can also simply delete the corresponding :program:`Cython` generated files. This will spare you some compilation time.
     
