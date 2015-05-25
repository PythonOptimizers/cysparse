..  cysparse_for_library_mainteners:

============================================
:program:`CySparse` for library maintainers
============================================

..  warning:: TO REWRITE COMPLETELY

The main difficulty to maintain the :program:`CySparse` library is to understand and master the automatic code generation from templated source code. We use the template engine :program:`Jinja2` and some hard coded 
conventions. We explain and justify these conventions in the following sections.  

The :class:`LLSparseMatrix` class in details
===============================================

All matrix classes are straightforward with perhaps the only exception of the :class:`LLSparseMatrix` class. This class is really a container where you can easily add or remove elements but it is not much more than that: a container. As such, it is **not** optimized for any matrix operation. There are some internal optimisations though to add or retrieve an element and we explain them here.

Elements are linked in chains (aka linked lists)
--------------------------------------------------

The :class:`LLSparseMatrix` container is made of 4 one dimensionnal arrays and one pointer (``free``):

- ``root[i]``: points to the first elements of row ``i``;
- ``col[k]``: contains the column index ``j`` of an element stored at the ``k``-th place;
- ``val[k]``: contains the value of the element stored at the ``k``-th place;
- ``link[k]``: contains the pointer to the next element in the chain where the ``k``-th element belongs.

The pointer ``free`` points to the first free element in the chain of free elements.

Despite its name, *linked list* sparse matrix, this container doesn't use list of pointers but **only** arrays of indices. "Pointers" are indices in arrays and to denote a pointer to ``NULL`` the value ``-1`` if used. For instance, if ``free == -1``,
this means that the chain of free elements is empty.

Then chain can be traversed by following the ``link`` array:

..  image:: images/ll_mat_link.*
    :width: 400 pt

Two chains are depicted in the picture. First, the chain with free elements. These are elements that where removed. The ``free`` pointer points to the first element of this chain and ``link[free]`` if :math:`f_0` which points to 
the second element in this chain. Whenever a new elements is added, it will take the place of this first element.

The second chain starts with element (:math:`j_0`, :math:`v_0`, :math:`k_0`). :math:`k_0` points to the second element (:math:`j_1`, :math:`v_1`, :math:`k_1`) and :math:`k_1` points to the second and last element (:math:`j_2`, :math:`v_2`, :math:`k_2`).
 
Inside the arrays, the elements can be stored in any order.

Elements are "aligned" row wise (chains correspond to rows)
---------------------------------------------------------------

Each chain of elements corresponds to one row of the matrix. The pointer ``root[i]`` points to the first element of the ``i`` :sup:`th` row. If the above chain (:math:`k_0`, :math:`k_1` and :math:`k_2` ) correspond to the only 
elements on row ``i``, ``root[i]`` would point to element (:math:`j_0`, :math:`v_0`, :math:`k_0`). If row ``i`` doesn't have any element, ``root[i] == -1``.

To traverse the ``i``:sup:`th` row, simply use:

..  code-block:: python

    k = self.root[i]
    
    while k != -1:
        # we consider element A[i, j] == val
        j = self.col[k]
        val = self.val[k]
        ...
        k = self.link[k]


Inside a row, elements are ordered by column order
----------------------------------------------------

If the chain corresponding to row ``i`` is :math:`k_0, k_1, \ldots, k_p`, then we know that the corresponding column indices are ordered: :math:`j_0 < j_1 < \ldots < j_p`. When an element is added with the ``put(i, j, val)`` method, this new element is inserted in the right place, swapping pointers elements of ``link`` if necessary.

This means that looking for an element ``A[i, k]``, one can simply use:

..  code-block:: python

    k = self.root[i]
    
    while k != -1:
        
        if self.col[k] > j:
            # element doesn't exist
            break
        
        if self.col[k] == j:
            # element exists
            ...
        
        k = self.link[k]




Meta-programming aka code generation
=====================================

:program:`CySparse` allows the use of different types at run time and most typed classes comes in different typed flavours. This feature comes with a price. Because we wanted to write the library completely 
in :program:`Cython`, we decided to go for the explicit template route, i.e. we write **templated source code** and and use **explicit names** in the source code.
This automatic generation process ask for some rigour and takes some time to master. If you follow the next conventions stricly, you should be fine. If you don't follow them then probably the code won't even compile or 
if it does you might generate difficult to find bugs. Trust me on this one.

..  warning:: Follow the conventions stricly to write templated source code.

Justifications
-----------------

Following conventions is not always easy, especially if you don't understand them. In this sub-section we try to convince you or at least we try to explain and justify some choices I (Nikolaj) made.

These conventions were made with the following purpose in mind:

- respect the DRY (Don't Repear Yourself) principle;
- if the conventions are not followed, the code shouldn't compile;
- prefer static over dynamic dispatch;
- use typed variables whenever possible;
- keep the code simple whenever it doesn't sacrifice to efficiency even if the solutions are not Pythonesque;

We develop these key ideas in the following sub-sections.

Respect the DRY principle
^^^^^^^^^^^^^^^^^^^^^^^^^^

Don't write the same code twice. This means of course than whenever you can factorize some common code, you should do so but in our case, because we lack the notion of *templates* (like :program:`C++` templates), we 
**have** to repeat ourselves and rewrite the classes with different types. This is the main reason to use a template engine and templated code.  

If the conventions are not respected, the code shouldn't compile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enforce the use of the conventions, we try to enforce them by the compiler (whether the :program:`C`, the :program:`Cython` or :program:`Python` compiler). Often, you'll find that templated code have guards to ensure that 
types are recognized and otherwise to generate garbish that won't compile.

The name convention is written explicitely: if you don't respect it, you won't be able to use the :program:`generate_code.py` script. This is on purpose.

Prefer static over dynamic dispatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Even if :program:`Python` is a dynamic language, efficient :program:`Cython` code **needs** typing. This typing can be done dynamically with long and tedious ``if/then`` combinations or we can let the compiler 
do the dispatch in our place at compile time whenever possible. This is the main reason why there are as many ``LLSparseMatrixView`` classes as there are ``LLSparseMatrix`` classes. Strictly speaking, we don't need 
more ``LLSparseMatrixView`` classes than the number of index types but then you need to dynamically dispatch some operations like the creation of a corresponding ``

Use typed variables whenever possible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:program:`Cython` really shines when it can deduce some static typing, especially in numeric loops. Therefor try to type variables **if** you know their type in advance [#typed_variables]_.


Our hope is to keep a nice balance between the difficulty of coding and the easiness to maintain the code. When generating automatically code, these two don't necessarily go hand in hand. 

If you find some code that doesn't follow these conventions, report it or even better change it!

Types
------



Basic types
^^^^^^^^^^^^^^^

For different reasons [#use_C99_quick_justification]_ (???)

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


Two categories of types
^^^^^^^^^^^^^^^^^^^^^^^^

We allow the use of different types at two levels:

- for the indices (``INT32_t`` and ``INT64_t``) [#signed_vs_unsigned_integers]_;
- for the matrix elements (**all** the basic types).



Add (or remove) a new type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conventions
-----------

File names and directories
^^^^^^^^^^^^^^^^^^^^^^^^^^^
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


:program:`Jinja2` conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Automatic generation scripts
------------------------------

**All** generated files can be generated by invoking a **single** script: 

..  code-block:: bash

    python generate_code.py


..  raw:: html

    <h4>Footnotes</h4>
    
..  [#typed_variables] Use your intelligence and knowledge of :program:`Cython`. Know when it makes a difference to type a variable.

..  [#use_C99_quick_justification] we use :program:`C99` for its superiority compared to :program:`ANSI C` (:program:`C89` or :program:`C90` which is the same). Among others:
    
        - the INFINITY and NAN macros;
        - its complex types;
        - inline functions;
        
..  [#signed_vs_unsigned_integers] We don't want to enter into the debate unsigned vs signed integers. Accept this as a fact. Beside, we use internally negative indices.
