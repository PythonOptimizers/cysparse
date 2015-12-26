"""
Very basic test unit generator.

"""
import sys
from collections import OrderedDict


# Copied from generate_code.py: any better idea?
# For tests
MATRIX_CLASSES = OrderedDict()
MATRIX_CLASSES['LLSparseMatrix'] = 'll_mat_matrices.ll_mat'
MATRIX_CLASSES['CSCSparseMatrix'] = 'csc_mat_matrices.csc_mat'
MATRIX_CLASSES['CSRSparseMatrix'] = 'csr_mat_matrices.csr_mat'

MATRIX_VIEW_CLASSES = OrderedDict()
MATRIX_VIEW_CLASSES['LLSparseMatrixView'] = 'll_mat_views.ll_mat_views'

MATRIX_PROXY_CLASSES = OrderedDict()
MATRIX_PROXY_CLASSES['TransposedSparseMatrix'] = 'sparse_proxies.t_mat'
MATRIX_PROXY_CLASSES['ConjugatedSparseMatrix'] = 'sparse_proxies.complex_generic.conj_mat'
MATRIX_PROXY_CLASSES['ConjugateTransposedSparseMatrix'] ='sparse_proxies.complex_generic.h_mat'


MATRIX_LIKE_CLASSES = OrderedDict()
MATRIX_LIKE_CLASSES.update(MATRIX_CLASSES)
MATRIX_LIKE_CLASSES.update(MATRIX_PROXY_CLASSES)

ALL_SPARSE_OBJECT = OrderedDict()
ALL_SPARSE_OBJECT.update(MATRIX_CLASSES)
ALL_SPARSE_OBJECT.update(MATRIX_PROXY_CLASSES)
ALL_SPARSE_OBJECT.update(MATRIX_VIEW_CLASSES)

TEST_TYPES = ['matrices', 'matrix-likes', 'sparse-likes']

TEST_FILE_PROLOGUE = '''"""
This file tests XXX for all %s objects.

"""
'''

TEST_FILE_BIG_SEPARATOR = '''
########################################################################################################################
# %s
########################################################################################################################
'''

TEST_FILE_MEDIUM_SEPARATOR = '''
#######################################################################
# %s
#######################################################################
'''

TEST_FILE_SMALL_SEPARATOR = '''
#=======================================================
# %s
'''

TEST_FILE_EPILOGUE = '''
if __name__ == '__main__':
    unittest.main()
'''

class TestGenerator(object):

    def __init__(self, test_name, test_type=None):
        super(TestGenerator, self).__init__()

        self.test_name = test_name

        self.test_type = None
        self.test_type_dict = None
        if test_type is not None:
            assert test_type in TEST_TYPES, "The type of test ('%s') is not recognized" % test_type
            self.test_type = test_type
            self.find_type_dict()

    def find_type_dict(self):
        if self.test_type == 'matrices':
            self.test_type_dict = MATRIX_CLASSES
        elif self.test_type == 'matrix-likes':
            self.test_type_dict = MATRIX_LIKE_CLASSES
        elif self.test_type == 'sparse-likes':
            self.test_type_dict = ALL_SPARSE_OBJECT
        else:
            raise TypeError("Test type '%s' is not recognized" % self.test_type)

    def generate_class_cases(self, OUT):
        len_dict = len(self.test_type_dict)
        for i, class_name in enumerate(self.test_type_dict.keys()):
            if i == 0:
                OUT.write(u"{%s if class == '%s' %s}\n\n" % ('%', class_name, '%'))
            else:
                OUT.write(u"{%s elif class == '%s' %s}\n\n" % ('%', class_name, '%'))

        OUT.write(u"{%s else %s}\n" % ('%', '%'))
        OUT.write("YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE\n")
        OUT.write(u"{%s endif %s}\n\n" % ('%', '%'))

    def generate_class_name(self, case):
        """
        Return a unit test class name.
        """
        return "class CySparse" + self.test_name + case + "_@class@_@index@_@type@_TestCase(unittest.TestCase)"

    def generate_test(self, test_type=None, OUTSTREAM=sys.stdout):
        """
        Generate one test file add pass it to a output stream.

        Args:
            test_type: Type of test. No need to specify it here if you have specify it in the constructor.
            OUTSTREAM: Output stream. ``sys.stdout`` by default.

        Returns:
            A string with the content of a test file.

        """
        if test_type is not None:
            assert test_type in TEST_TYPES, "The type of test ('%s') is not recognized" % test_type
            self.test_type = test_type
            self.find_type_dict()

        assert self.test_type is not None, "The test type cannot be None"
        assert self.test_type_dict is not None, "The test type dict cannot be None"

        OUTSTREAM.write('#!/usr/bin/env python')
        OUTSTREAM.write('\n')
        OUTSTREAM.write('\n')
        OUTSTREAM.write(TEST_FILE_PROLOGUE % self.test_type)
        OUTSTREAM.write('\n')

        OUTSTREAM.write('import unittest\n')
        OUTSTREAM.write('from cysparse.sparse.ll_mat import *\n')
        OUTSTREAM.write('from cysparse.common_types.cysparse_types import *\n\n')

        OUTSTREAM.write(TEST_FILE_BIG_SEPARATOR % 'Tests')
        OUTSTREAM.write('\n')

        # case 1
        OUTSTREAM.write(TEST_FILE_MEDIUM_SEPARATOR % 'Case: store_symmetry == False, Store_zero==False')
        OUTSTREAM.write(self.generate_class_name('NoSymmetryNoZero'))
        OUTSTREAM.write(':\n')
        OUTSTREAM.write('    def setUp(self):\n\n')
        self.generate_class_cases(OUTSTREAM)
        OUTSTREAM.write('    def test_XXX(self):\n\n')

        # case 2
        OUTSTREAM.write(TEST_FILE_MEDIUM_SEPARATOR % 'Case: store_symmetry == True, Store_zero==False')
        OUTSTREAM.write(self.generate_class_name('WithSymmetryNoZero'))
        OUTSTREAM.write(':\n')
        OUTSTREAM.write('    def setUp(self):\n\n')
        self.generate_class_cases(OUTSTREAM)
        OUTSTREAM.write('    def test_XXX(self):\n\n')

        # case 3
        OUTSTREAM.write(TEST_FILE_MEDIUM_SEPARATOR % 'Case: store_symmetry == False, Store_zero==True')
        OUTSTREAM.write(self.generate_class_name('NoSymmetrySWithZero'))
        OUTSTREAM.write(':\n')
        OUTSTREAM.write('    def setUp(self):\n\n')
        self.generate_class_cases(OUTSTREAM)
        OUTSTREAM.write('    def test_XXX(self):\n\n')

        # case 4
        OUTSTREAM.write(TEST_FILE_MEDIUM_SEPARATOR % 'Case: store_symmetry == True, Store_zero==True')
        OUTSTREAM.write(self.generate_class_name('WithSymmetrySWithZero'))
        OUTSTREAM.write(':\n')
        OUTSTREAM.write('    def setUp(self):\n\n')
        self.generate_class_cases(OUTSTREAM)
        OUTSTREAM.write('    def test_XXX(self):\n\n')

        OUTSTREAM.write(TEST_FILE_EPILOGUE)
        OUTSTREAM.write('\n')



