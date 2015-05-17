..  cysparse_for_library_mainteners:

<<<<<<< HEAD
==========================================
:program:`CySparse` for library mainteners
==========================================
=======
============================================
:program:`CySparse` for library maintainers
============================================

..  warning:: TO REWRITE COMPLETELY

The main difficulty to maintain the :program:`CySparse` library is to understand and master the automatic generation of code from templated source code. We use the template engine :program:`Jinja2` and some hard coded 
conventions. 
>>>>>>> master

Meta-programming aka code generation
=====================================

<<<<<<< HEAD
We choose to generate some code by automatic generation. :program:`Python`, :program:`Cython` or :program:`C` are quite limited
when it comes to use generic types. Because we want to write this library in :program:`Cython`, we decided to generate automatically 
different files for different types used.

Types
------

Generic types
^^^^^^^^^^^^^^^

We use the following generic types:

- ``INT32_t``
- ``UINT32_t``
- ``INT64_t``
- ``UINT64_t``
- ``FLOAT32_t``
- ``FLOAT64_t``
- ``COMPLEX64_t``
- ``COMPLEX128_t``

These types can be defined (``ctypedef``\ed) as any buildin or custom types (like a :program:`numpy` type or a custom ``struct`` or even a custom :program:`C++` class).
See the files :file:`xxx_types_xxx.pxd` in the directory :file:`cysparse/types` for some examples.

The names are quite explicit and we suppose that the real types used reflect that, i.e. an ``INT32_t`` has a smaller (or equal) size than a ``INT64_t``. We don't impose any other restriction
and while the numbers are explicit you shouldn't expect that that exact amount of bits are used.


The ``COMPLEX`` type
^^^^^^^^^^^^^^^^^^^^

The complex type needs some special attention. Whatever the ``class`` or ``struct`` used to define a complex, it **must** contain two fields (or attributes):

- ``real``
- ``imag``

that should be publicly accessible like this:

..  code-block:: cython

    COMPLEX128_t z
    z.real = ...
    z.imag = ...
    
Both fields **must** be of the exact same type. :program:`Python` offers such a ``struct`` but :program:`Cython` and :program:`C++` don't. Their versions of complex is a little bit different and even though it shares some similarities, you **cannot** access the fields ``real`` and ``imag`` directly. These types are thus **not** compatible with our implementation. We have defined our own complex type. See XXX.
=======
:program:`CySparse` allows the use of different types at run time and most typed classes comes in different typed flavours. This feature comes with a price. Because we wanted to write the library completely 
in :program:`Cython`, we decided to go for the explicit template route, i.e. templated source code is written and explicit names are used in the generated :program:`Cython` code.
This automatic generation process ask for some rigour and takes some time to master. If you follow the next conventions stricly, you should be fine. If you don't follow them then probably the code won't even compile or 
if it does you might generate difficult to find bugs. Trust me on this one.

..  warning:: Follow the conventions stricly to write templated source code.

Justifications
-----------------

Following conventions is not always easy, especially if you don't understand them. In this sub-section we try to convince you or at least we try to explain and justify some choices I (Nikolaj) made.

These conventions were made with the following purpose in mind:

- respect the DRY (Don't Repear Yourself) principle;
- if the conventions are not followed, the code shouldn't compile;
- prefer static dispatch instead of dynamic dispatch;
- use typed variables whenever possible;
- keep the code simple whenever it doesn't sacrifice to efficiency even if the solutions are not Pythonesque;

Respect the DRY principle
^^^^^^^^^^^^^^^^^^^^^^^^^^

Don't write the same code twice. This means of course than whenever you can factorize some common code, you should do so but in our case, because we lack the notion of *templates* (like :program:`C++` templates), we 
**have** to repeat ourselves and rewrite the classes with different types. This is the main reason to use a template engine and templated code.  

If the conventions are not respected, the code shouldn't compile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enforce the use of the conventions, we try to enforce them by the compiler (whether the :program:`C`, the :program:`Cython` or :program:`Python` compiler). Often, you'll find that templated code have guards to ensure that 
types are recognized and otherwise to generate garbish that won't compile.

The name convention is written explicitely: if you don't respect it, you won't be able to use the :program:`generate_code.py` script. This is on purpose.

Prefer static dispatch instead of dynamic dispatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Even if :program:`Python` is a dynamic language, efficient :program:`Cython` code **needs** typing. This typing can be done dynamically with long and tedious ``if/then`` combinations or we can let the compiler 
do the dispatch in our place at compile time whenever possible. This is the main reason why there are as many ``LLSparseMatrixView`` classes as there are ``LLSparseMatrix`` classes. Strictly speaking, we don't need 
more ``LLSparseMatrixView`` classes than the number of index types but then you need to dynamically dispatch some operations like the creation of a corresponding ``



Our hope is to keep a nice balance between the difficulty of coding and the easiness to maintain the code. When generating automatically code, these two don't necessarily go hand in hand. 

If you find some code that doesn't follow these conventions, report it or even better change it!

Types
------



Basic types
^^^^^^^^^^^^^^^

We use the following basic types:

==============================  ==============================
:program:`CySparse`             C99 types
==============================  ==============================
``INT32_t``                     ``int``
``UINT32_t``                    ``unsigned int``
``INT64_t``                     ``long``
``UINT64_t``                    ``unsigned long``
``FLOAT32_t``                   ``float``
``FLOAT64_t``                   ``double``
``COMPLEX64_t``                 ``float complex``
``COMPLEX128_t``                ``double complex``
==============================  ==============================



>>>>>>> master


Add (or remove) a new type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

<<<<<<< HEAD
Automatic
------------
=======
Automatic generation
------------------------
>>>>>>> master

**All** generated files can be generated by invoking a **single** script: 

..  code-block:: bash

    python generate_code.py

Conventions
-----------

To keep the generation of code source files as simple as possible, we follow some conventions. This list of conventions is **strict**: if you depart from these conventions, the code will **not** compile.

- **Don't** use fused types: this feature is too **experimental**.
- Template files have the following extensions:
    
  ============================= ============================= ==================================
  :program:`Cython`             :program:`CySparse` template  File type
  ============================= ============================= ==================================
  ``.pxd``                      ``.cpd``                      Definition files.
  ``.pyx``                      ``.cpx``                      Implementation files.
  ``.pxi``                      ``.cpi``                      Text files to insert verbatim.
  ============================= ============================= ==================================
  
  For python files:
  
  ============================= ============================= ==================================
  :program:`Python`             :program:`CySparse` template  File type
  ============================= ============================= ==================================
  ``.py``                       ``.cpy``                      Python module files.
  ============================= ============================= ==================================
  

- Any *template* directory must **only** contain the template files and the generated files. This is because
  all files with the right extension are considered as templates and all the other files are considered as generated 
  (and can be thus automatically erased). This clear distinction allows also to have a strict separation between 
  automatically generated files and the rest of the code.
- Index types are replaced whenever the variable ``@index@`` is encountered, Element types are replaced whenever the variable ``@type@`` is encountered.
- Generated **file names**:

  - for a file ``my_file.cpx`` where we only replace an index type ``INT32_t``: ``my_file_INT32_t.pyx``;
  - for a file ``my_file.cpx`` where we replace an index type ``INT32_t`` **and** an elment type ``FLOAT64_t``: ``my_file_INT32_t_FLOAT_t.pyx``.
    
- Generated **class/method/function names**:



    
