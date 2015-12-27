..  cysparse_intallation:

===================================
Installation
===================================

There are basically [#tricky_installations]_, two modes to install :program:`CySparse`:

- :program:`Python` mode and
- :program:`Cython` mode.

In :program:`Python` mode, you install the library as a usual :program:`Python` library. In :program:`Cython` mode a little bit more work is involved as you also need to generate the source code from templated files.

..  ###########################################################################################################################
    ##### PYTHON INSTALLATION
    ###########################################################################################################################
    
:program:`Python` installation
===================================

The installation is done in a few simple steps:

1. Clone the repository;
2. Install the dependencies;
3. Copy :file:`cysparse_template.cfg` to :file:`cysparse.cfg`;
4. Compile and install the library:

We detail these steps in the next sections.

Clone the repository
---------------------

Start by cloning the :program:`GitHub` repository:

..  code-block:: bash

    git clone https://github.com/PythonOptimizers/cysparse

Beware that you'll need a good internet connexion: the code weight a little bit more than ... 200MB!

Install the depencies
----------------------

- :program:`NumPy`;
- :program:`configparser` (**not** :program:`ConfigParser`);

[TO BE WRITTEN]


All :program:`Python` dependencies are described in the :file:`requirements.txt` files. You can easily install them all with:

..  code-block:: bash

    pip install -r requirements.txt

or a similar command. 

Compile and install the library
---------------------------------

The preferred way to install the library is to install it in its own `virtualenv`.

Wheter using a virtual environment or not, use the traditionnal:

..  code-block:: bash

    python setup.py install

to compile and install the library.

..  ###########################################################################################################################
    ##### CYTHON INSTALLATION
    ###########################################################################################################################
    
:program:`Cython` installation
===================================

The installation is done in a few simple steps:

1. Clone the repository;
2. Install the dependencies;
3. Tweak the configuration file :file:`cysparse.cfg`;
4. Generate the source code;
5. Compile and install the library:

We detail these steps in the next sections.

Clone the repository
---------------------

Start by cloning the :program:`GitHub` repository:

..  code-block:: bash

    git clone https://github.com/PythonOptimizers/cysparse

Beware that you'll need a good internet connexion: the code weight a little bit more than ... 200MB!

Install the dependencies
--------------------------

:program:`CySparse`
""""""""""""""""""""

- :program:`Cython`
- :program:`Jinja2`
- argparse
- fortranformat


..  _cysparse_configuration_file:

Tweak the configuration file :file:`cysparse.cfg`
---------------------------------------------------

The :file:`cysparse.cfg` is written for :program:`Python`'s ``ConfigParser`` module. Section stars with ``[my_section]`` followed by one or several lines where a variable on the left is assigned a value on the right. 
An equal sign (``=``) between  the variable and the value defines the assignment. Here is an example:

..  code-block:: text

    [CODE_GENERATION]
    # log file name **without** extension (by default, we use '.log')
    log_name = cysparse_generate_code
    # DEBUG/INFO/WARNING/ERROR/CRITICAL
    log_level = INFO

All what follows an ``#`` is considered as comments. In the example above, we have a section ``CODE_GENERATION`` and two variables ``log_name`` and ``log_level``.

Logging
""""""""

By default, we log the code/tests generation process. The logging is send to a file and to the screen. You can provide a filename with the ``log_name`` variable (this is both used for the logging engine name and the file name) and
the log level by defining the ``log_level`` variable. The file has ``.log`` as extension. The log level values are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` and ``CRITICAL``. You can also choose log level for the file and the console with ``console_log_level`` 
and ``file_log_level``. For instance:

..  code-block:: text

    log_name = cysparse_generate_code

    log_level = INFO
    console_log_level = CRITICAL
    file_log_level = WARNING

means that you use a log file names ``cysparse_generate_code.log``

32bits/64bits platforms 
""""""""""""""""""""""""""""""""""""""""""""""

By default, :program:`CySparse` tests if your platform is 32bits or 64bits [#other_platforms]_ and use (``INT32_t``, ``INT64_t``) or (``INT64_t``, ``FLOAT64_t``) for the index and element types respectively.
You can change/force this behavior by setting ``DEFAULT_INDEX_TYPE`` and ``DEFAULT_ELEMENT_TYPE`` to ``32bits`` or ``64bits``.

Generate the source code
--------------------------


Some parts of the library source code have to be generated **if** you use :program:`Cython` or wish to generate the code from scratch. We use a script:

..  code-block:: bash

	python generate_code.py -r cysparse
    
If you need help, try the ``-h`` switch. If you intend to modify the source code, we invite you to read the developers manual.

Compile and install the library
---------------------------------

The preferred way to install the library is to install it in its own `virtualenv`.

Wheter using a virtual environment or not, use the traditionnal:

..  code-block:: bash

    python setup.py install

to compile and install the library.

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



Further dependencies
========================



Documentation
------------------

- :program:`Sphinx`
- sphinx-bootstrap-theme

Unit testing
-----------------

- :program:`PySparse`

Performance testing
---------------------

- :program:`PySparse`
- benchmark.py (https://github.com/optimizers/benchmark.py)







..  raw:: html

    <h4>Footnotes</h4>
    
..  [#tricky_installations] Some special configurations might need a complete or partial :program:`Cython` source generation.

..  [#other_platforms] For other platforms, have a look at the file :file:`cysparse/common_types/cysparse_types.pxd`.

..  [#cython_try_recompiling] The problem is interdependencies between source files that are not catched at compile time. Whenever :program:`Cython` can catch them at runtime, it throws this ``ValueError``.

..  [#missing_dependencies_generated_templates] Interdependencies between generated templates are **not** monitored. Instead of recompiling everything from scratch, you can also simply delete the corresponding :program:`Cython` generated files. This will spare you some compilation time.
     
