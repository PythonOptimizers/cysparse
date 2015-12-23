..  cysparse_intallation:

===================================
:program:`CySparse` installation
===================================

There are basically [#tricky_installations]_, two modes to install :program:`CySparse`:

- :program:`Python` mode and
- :program:`Cython` mode.

In :program:`Python` mode, you install the library as a usual :program:`Python` library. In :program:`Cython` mode a little bit more work is involved as you also need to generate the source code from templated files.


The installation is done in a few simple steps:

1. Clone the repository;
2. Install the dependencies;
3. Tweak the configuration file :file:`cysparse.cfg`;
4. Generate the source code if needed (i.e. in :program:`Cython` mode);
5. Compile and install the library:

We detail these steps in the next sections for both installation modes.

:program:`Python` installation mode
===================================

Clone the repository
---------------------



Install the depencies
----------------------

All :program:`Python` dependencies are described in the :file:`requirements.txt` files. You can easily install them all with:

..  code-block:: bash

    pip install -r requirements.txt

or a similar command. Other dependencies need some manual installation. Read further.


:program:`CySparse`
""""""""""""""""""""

- :program:`Cython`
- :program:`Jinja2`
- argparse
- fortranformat
- :program:`SuiteSparse` (for the moment, it not possible to install :program:`CySparse` **without** :program:`SuiteSparse`)

Documentation
""""""""""""""""

- :program:`Sphinx`
- sphinx-bootstrap-theme

Unit testing
"""""""""""""""

- :program:`PySparse`

Performance testing
"""""""""""""""""""""""

- :program:`PySparse`
- benchmark.py (https://github.com/optimizers/benchmark.py)


Tweak the configuration file :file:`cysparse.cfg`
---------------------------------------------------

[THIS IS WORK IN PROGRESS]

# log file name **without** extension (by default, we use '.log')
log_name = cysparse_generate_code
# DEBUG/INFO/WARNING/ERROR/CRITICAL
log_level = INFO
console_log_level = WARNING
file_log_level = WARNING


# 32bits/64bits
# if left blank, we use INT64_t on 64 bits platforms and INT32_t on 32 bits platforms
DEFAULT_INDEX_TYPE =

Generate the source code
--------------------------


Some parts of the library source code have to be generated **if** you use :program:`Cython` or wish to generate the code from scratch. We use a script:

..  code-block:: bash

	python generate_code.py -a
    
The switch ``-a`` stands for ``--all`` and generates the entire library. If you need help, try the ``-h`` switch.

Compile and install the library
---------------------------------

The preferred way to install the library is to install it in its own `virtualenv`.

Wheter using a virtual environment or not, use the traditionnal:

..  code-block:: bash

    python setup.py install

to compile and install the library.

:program:`Cython` installation mode
===================================


Inconveniences
----------------

- Sometimes :program:`Cython` can ask for a complete recompilation. 
  Whenever this happens, it displays the following message when trying to import the library 
  into :program:`Python`:

  ..  code-block:: bash

      ValueError: XXX has the wrong size, try recompiling

  where XXX is the first class that has the wrong size. The easiest way to deal with this is to recompile all the .pyx files again (you can force this by removing
  all the .c files) [#cython_try_recompiling]_.

  See Robert Bradshaw's `answer <https://groups.google.com/forum/?hl=en#!topic/cython-users/cOAVM0whJkY>`_. 
  See also `enhancements distutils_preprocessing <https://github.com/cython/cython/wiki/enhancements-distutils_preprocessing>`_.

- **If** you modify the templated code, some dependencies might be missing in the (generated) ``setup.py`` file and require manual intervention, 
  i.e. recompilation. The easiest way to go is to recompile everything from scratch [#missing_dependencies_generated_templates]_. First delete the generated files:

  ..  code-block:: bash

      python generate_code.py -ac
        
  where ``-ac`` stands for ``a``\ll and ``c``\lean. This will delete **all** generated ``.pxi``, ``.pxd`` and ``.pyx`` :program:`Cython` files. Then delete the generated :program:`C` files:

  ..  code-block:: bash

      python clean.py
        
  This will delete **all** :program:`C` ``.c`` files. You can then recompile the library from scratch.


..  raw:: html

    <h4>Footnotes</h4>
    
..  [#tricky_installations] Some special configurations might need a complete or partial :program:`Cython` source generation.

..  [#cython_try_recompiling] The problem is interdependencies between source files that are not catched at compile time. Whenever :program:`Cython` can catch them at runtime, it throws this ``ValueError``.

..  [#missing_dependencies_generated_templates] Interdependencies between generated templates are **not** monitored. Instead of recompiling everything from scratch, you can also simply delete the corresponding :program:`Cython` generated files. This will spare you some compilation time.
     
