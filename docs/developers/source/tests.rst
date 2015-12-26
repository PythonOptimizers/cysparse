..  _unittests:

=====================
Tests
=====================


``unittest`` generator
========================

In the directory ``tests``, you can find a directory ``generator``. Inside you'll find some :program:`Python` scripts to generate **templates** of ``unittest`` test files. You can then add manually you tests and
use :program:`cygenja` to generate the :program:`Python` ``unittest`` files.

These scripts are very basic but I (Nikolaj)
found them quite handy. Not only do they unify the test files, they also allow you to test **every** possible cases quite rapidly. At the moment of writing, the four possible cases concern the storage schemes:

- with/without the storage of symmetry (``store_symmetry`` set to ``True/False``);
- with/without the explicit storage of zeros (``store_zero`` set to ``True/False``).

To use the ``generate_unit_t_file.py`` script, invoke:

..  code-block:: bash

    python generate_unit_t_file.py -t X -n MyTestCaseClassName
    
where ``X`` is an integer between 0 and 2 and ``MyTestCaseClassName`` is the name of the ``unittest`` classes **and** the name of the test file generated.
Is is better to use **CapitalizedWords** for the name arguement.

The ``X`` integer can have the following values:

- ``0``: matrices test cases: all real matrix classes(``LLSparseMatrix``, ``CSCSparseMatrix`` and ``CSRSparseMatrix``);
- ``1``: matrix-like test cases: all real matrices (case ``0`` above) and all proxy classes (``TransposedSparseMatrix``, ``ConjugatedSparseMatrix`` and ``ConjugateTransposedSparseMatrix``);
- ``2``: sparse-like test cases: all matrix-like objects (case ``1`` above) and all view classes (``LLSparseMatrixView``).

The invocation above with ``X`` set to ``2`` replacing ``MyTestCaseClassName`` by ``Copy`` , will result in a similar :file:`test_copy.cpy` template file:

..  only:: html

    ..  code-block:: jinja

        #!/usr/bin/env python

        """
        This file tests XXX for all sparse-likes objects.

        """

        import unittest


        ########################################################################################################################
        # Tests
        ########################################################################################################################


        #######################################################################
        # Case: store_symmetry == False, Store_zero==False
        #######################################################################
        class CySparseCopyNoSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
            def setUp(self):

        {% if class == 'LLSparseMatrix' %}

        {% elif class == 'CSCSparseMatrix' %}

        {% elif class == 'CSRSparseMatrix' %}

        {% elif class == 'TransposedSparseMatrix' %}

        {% elif class == 'ConjugatedSparseMatrix' %}

        {% elif class == 'ConjugateTransposedSparseMatrix' %}

        {% elif class == 'LLSparseMatrixView' %}

        {% else %}
        YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
        {% endif %}

            def test_XXX(self):


        #######################################################################
        # Case: store_symmetry == True, Store_zero==False
        #######################################################################
        class CySparseCopyWithSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
            def setUp(self):

        {% if class == 'LLSparseMatrix' %}

        {% elif class == 'CSCSparseMatrix' %}

        {% elif class == 'CSRSparseMatrix' %}

        {% elif class == 'TransposedSparseMatrix' %}

        {% elif class == 'ConjugatedSparseMatrix' %}

        {% elif class == 'ConjugateTransposedSparseMatrix' %}

        {% elif class == 'LLSparseMatrixView' %}

        {% else %}
        YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
        {% endif %}

            def test_XXX(self):


        #######################################################################
        # Case: store_symmetry == False, Store_zero==True
        #######################################################################
        class CySparseCopyNoSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
            def setUp(self):

        {% if class == 'LLSparseMatrix' %}

        {% elif class == 'CSCSparseMatrix' %}

        {% elif class == 'CSRSparseMatrix' %}

        {% elif class == 'TransposedSparseMatrix' %}

        {% elif class == 'ConjugatedSparseMatrix' %}

        {% elif class == 'ConjugateTransposedSparseMatrix' %}

        {% elif class == 'LLSparseMatrixView' %}

        {% else %}
        YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
        {% endif %}

            def test_XXX(self):


        #######################################################################
        # Case: store_symmetry == True, Store_zero==True
        #######################################################################
        class CySparseCopyWithSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
            def setUp(self):

        {% if class == 'LLSparseMatrix' %}

        {% elif class == 'CSCSparseMatrix' %}

        {% elif class == 'CSRSparseMatrix' %}

        {% elif class == 'TransposedSparseMatrix' %}

        {% elif class == 'ConjugatedSparseMatrix' %}

        {% elif class == 'ConjugateTransposedSparseMatrix' %}

        {% elif class == 'LLSparseMatrixView' %}

        {% else %}
        YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
        {% endif %}

            def test_XXX(self):


        if __name__ == '__main__':
            unittest.main()

..  only:: latex

    ..  code-block:: jinja

        #!/usr/bin/env python

        """
        This file tests XXX for all sparse-likes objects.

        """

        import unittest


        #######################################################################
        # Tests
        #######################################################################


        #######################################################################
        # Case: store_symmetry == False, Store_zero==False
        #######################################################################
        class CySparseCopyNoSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
            def setUp(self):

        {% if class == 'LLSparseMatrix' %}

        {% elif class == 'CSCSparseMatrix' %}

        {% elif class == 'CSRSparseMatrix' %}

        {% elif class == 'TransposedSparseMatrix' %}

        {% elif class == 'ConjugatedSparseMatrix' %}

        {% elif class == 'ConjugateTransposedSparseMatrix' %}

        {% elif class == 'LLSparseMatrixView' %}

        {% else %}
        YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
        {% endif %}

            def test_XXX(self):


        #######################################################################
        # Case: store_symmetry == True, Store_zero==False
        #######################################################################
        class CySparseCopyWithSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
            def setUp(self):

        {% if class == 'LLSparseMatrix' %}

        {% elif class == 'CSCSparseMatrix' %}

        {% elif class == 'CSRSparseMatrix' %}

        {% elif class == 'TransposedSparseMatrix' %}

        {% elif class == 'ConjugatedSparseMatrix' %}

        {% elif class == 'ConjugateTransposedSparseMatrix' %}

        {% elif class == 'LLSparseMatrixView' %}

        {% else %}
        YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
        {% endif %}

            def test_XXX(self):


        #######################################################################
        # Case: store_symmetry == False, Store_zero==True
        #######################################################################
        class CySparseCopyNoSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
            def setUp(self):

        {% if class == 'LLSparseMatrix' %}

        {% elif class == 'CSCSparseMatrix' %}

        {% elif class == 'CSRSparseMatrix' %}

        {% elif class == 'TransposedSparseMatrix' %}

        {% elif class == 'ConjugatedSparseMatrix' %}

        {% elif class == 'ConjugateTransposedSparseMatrix' %}

        {% elif class == 'LLSparseMatrixView' %}

        {% else %}
        YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
        {% endif %}

            def test_XXX(self):


        #######################################################################
        # Case: store_symmetry == True, Store_zero==True
        #######################################################################
        class CySparseCopyWithSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
            def setUp(self):

        {% if class == 'LLSparseMatrix' %}

        {% elif class == 'CSCSparseMatrix' %}

        {% elif class == 'CSRSparseMatrix' %}

        {% elif class == 'TransposedSparseMatrix' %}

        {% elif class == 'ConjugatedSparseMatrix' %}

        {% elif class == 'ConjugateTransposedSparseMatrix' %}

        {% elif class == 'LLSparseMatrixView' %}

        {% else %}
        YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
        {% endif %}

            def test_XXX(self):


        if __name__ == '__main__':
            unittest.main()


which is our canvas for ``unittest`` files.

