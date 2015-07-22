#!/usr/bin/env python
#################################################################################################
# This script generates all templated code for CySparse
# It this the single one script to use before Cythonizing the CySparse library.
# This script is NOT automatically called by setup.py
#
# The order of code generation is from the "inside towards the outside":
#
# - first generate the most inner code, i.e. the code that is used inside other code;
# - layer by layer, generate the code that only depends on already action code.
#
# We use this homemade script with the Jinja2 template engine:
# http://jinja.pocoo.org/docs/dev/
#
#################################################################################################

from setup.files_finder import find_files

import os
import sys
import glob
import fnmatch

import argparse
import logging
import ConfigParser

from subprocess import call

from jinja2 import Environment, FileSystemLoader


#################################################################################################
# INIT
#################################################################################################
PATH = os.path.dirname(os.path.abspath(__file__))

LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }


def make_parser():
    """
    Create a comment line argument parser.

    Returns:
        The command line parser.
    """
    parser = argparse.ArgumentParser(description='%s: a Cython code generator for the CySparse library.' % os.path.basename(sys.argv[0]))
    parser.add_argument("-a", "--all", help="Create all action files.", action='store_true', required=False)

    parser.add_argument("-m", "--matrices", help="Create sparse matrices.", action='store_true', required=False)
    parser.add_argument("-s", "--setup", help="Create setup file.", action='store_true', required=False)
    parser.add_argument("-g", "--generic_types", help="Create generic types.", action='store_true', required=False)
    parser.add_argument("-l", "--linalg", help="Create Linear Algebra contexts.", action='store_true', required=False)
    parser.add_argument("-t", "--tests", help="Create generic tests.", action='store_true', required=False)
    parser.add_argument("-c", "--clean", help="Clean action files.", action='store_true', required=False)

    return parser


#################################################################################################
# JINJA2 FILTERS
#################################################################################################
####################################
# CYSPARSE/NUMPY TYPES
####################################
def type2enum(type_name):
    """
    Transform a real :program:`CySparse` type into the equivalent :program:`CySparse` enum type.

    For instance:

        INT32_t -> INT32_T

    Args:
        cysparse_type:

    """
    enum_name = type_name[:-1]
    enum_name = enum_name + type_name[-1].upper()

    return enum_name

# OLD VERSION
# def cysparse_type_to_numpy_c_type_with_t(cysparse_type):
#     """
#     Transform a :program:`CySparse` enum type into the corresponding :program:`NumPy` C-type.
#
#     For instance:
#
#         INT32_T -> int32_t
#
#     Args:
#         cysparse_type:
#
#     """
#     return cysparse_type.lower()


# NEW VERSION
def cysparse_type_to_numpy_c_type(cysparse_type):
    """
    Transform a :program:`CySparse` enum type into the corresponding :program:`NumPy` C-type.

    For instance:

        INT32_T -> npy_int32

    Args:
        cysparse_type:

    """
    return 'npy_' + str(cysparse_type.lower()[:-2])


def cysparse_type_to_numpy_type(cysparse_type):
    """
    Transform a :program:`CySparse` enum type into the corresponding :program:`NumPy` type.

    For instance:

        INT32_T -> int32

    Args:
        cysparse_type:

    """
    return cysparse_type.lower()[:-2]


def cysparse_type_to_numpy_enum_type(cysparse_type):
    """
    Transform a :program:`Cysparse` enum type into the corresponding :program:`NumPy` enum.

    For instance:

        FLOAT64_T -> NPY_FLOAT64

    Args:
        cysparse_type:
    """
    return 'NPY_' + cysparse_type.upper()[:-2]


def cysparse_type_to_real_sum_cysparse_type(cysparse_type):
    """
    Returns the best **real** type for a **real** sum for a given type.

    For instance:

        INT32_t -> FLOAT64_t

    Args:
        cysparse_type:

    """

    r_type = None

    if cysparse_type in ['INT32_t', 'UINT32_t', 'INT64_t', 'UINT64_t']:
        r_type = 'FLOAT64_t'
    elif cysparse_type in ['FLOAT32_t', 'FLOAT64_t']:
        r_type = 'FLOAT64_t'
    elif cysparse_type in ['FLOAT128_t']:
        r_type = 'FLOAT128_t'
    elif cysparse_type in ['COMPLEX64_t', 'COMPLEX128_t']:
        r_type = 'FLOAT64_t'
    elif cysparse_type in ['COMPLEX256_t']:
        r_type = 'FLOAT128_t'
    else:
        raise TypeError("Not a recognized type")

    assert r_type in ['FLOAT64_t', 'FLOAT128_t']

    return r_type


def cysparse_real_type_from_real_cysparse_complex_type(cysparse_type):
    """
    Returns the **real** type for the real or imaginary part of a **real** complex type.

    For instance:

        COMPLEX128_t -> FLOAT64_t

    Args:
        cysparse:

    """
    r_type = None

    if cysparse_type in ['COMPLEX64_t']:
        r_type = 'FLOAT32_t'
    elif cysparse_type in ['COMPLEX128_t']:
        r_type = 'FLOAT64_t'
    elif cysparse_type in ['COMPLEX256_t']:
        r_type = 'FLOAT128_t'
    else:
        raise TypeError("Not a recognized complex type")

    return r_type


####################################
# UMFPACK TYPES
####################################
def cysparse_real_type_to_umfpack_family(cysparse_type):
    if cysparse_type in ['INT32_t']:
        return 'i'
    elif cysparse_type in ['INT64_t']:
        return 'l'
    elif cysparse_type in ['FLOAT64_t']:
        return 'd'
    elif cysparse_type in ['COMPLEX128_t']:
        return 'z'
    else:
        raise TypeError("Not a recognized SuiteSparse Umfpack type")

####################################
# CHOLMOD TYPES
####################################
def cysparse_real_type_to_cholmod_prefix(cysparse_type):
    if cysparse_type in ['INT32_t']:
        return 'cholmod_'
    elif cysparse_type in ['INT64_t']:
        return 'cholmod_l'
    else:
        raise TypeError("Not a recognized SuiteSparse Cholmod type")


####################################
# MUMPS TYPES
####################################
def cysparse_real_type_to_mumps_family(cysparse_type):
    if cysparse_type in ['FLOAT32_t']:
        return 's'
    elif cysparse_type in ['FLOAT64_t']:
        return 'd'
    elif cysparse_type in ['COMPLEX64_t']:
        return 'c'
    elif cysparse_type in ['COMPLEX128_t']:
        return 'z'
    else:
        raise TypeError("Not a recognized Mumps type")


def clean_cython_files(logger, directory, file_list=None):
    """

    Args:
        logger:
        directory:
        file_list: File name **must** be absolute!

    Note:
        We don't test if ``directory`` and ``file_list`` correspond, i.e. we don't check if the absolute filenames
        contain ``directory`` as last subdirectory.
    """
    real_file_list = []

    logger.info("Cleaning directory '%s'" % directory)

    if file_list is not None:
        real_file_list.extend(file_list)
    else:
        real_file_list.extend(glob.glob(os.path.join(directory, '*.pxi')))
        real_file_list.extend(glob.glob(os.path.join(directory, '*.pxd')))
        real_file_list.extend(glob.glob(os.path.join(directory, '*.pyx')))

    for filename in real_file_list:
        try:
            os.remove(filename)
        except:
            logger.warning("Couln't remove file '%s'" % filename)


def render_template(template_filename, template_environment, context):
    return template_environment.get_template(template_filename).render(context)


def generate_template_files(logger, template_filenames, template_environment, context, ext):
    """
    Generate template file **if** the source file is more recent than the generated file.

    Args:
        logger:
        template_filenames:
        template_environment:
        context:
        ext:

    """
    for filename in template_filenames:
        logger.info("Parsing file '%s'" % filename)
        path, base_filename = os.path.split(filename)
        base_filename_without_extension = base_filename[:-4]

        code_rendered = render_template(filename, template_environment, context)
        code_filename = base_filename_without_extension + '%s' % ext
        code_filename_path = os.path.join(path, code_filename)

        # test if file is non existing or needs to be regenerated
        if not os.path.isfile(code_filename_path) or os.stat(filename).st_mtime - os.stat(code_filename_path).st_mtime > 1:

            with open(code_filename_path, 'w') as f:
                logger.info('   Generating file %s' % code_filename_path)
                f.write(code_rendered)
        else:
            logger.info('   Source didn\'t change')


def generate_following_only_index(logger, template_filenames, template_environment, template_context, index_types, ext):
    """
    Generate Cython code.

    Args:
        logger: logging engine.
        template_filenames: List of file names.
        environment: An Jinja2 Environment.
        index_types: List of index types.
        ext: Extension of the created files.

    Note:
        A file is created for each index types.
    """
    for filename in template_filenames:

        for index_name in index_types:

            context = {
                'index' : index_name,
                'index_list' : INDEX_TYPES
            }

            context.update(template_context)

            generate_template_files(logger, template_filenames, template_environment, context, '_%s%s' % (index_name, ext))


def generate_following_only_type(logger, template_filenames, template_environment, template_context, element_types, ext):
    """
    Generate Cython code.

    Args:
        logger: logging engine.
        template_filenames: List of file names.
        environment: An Jinja2 Environment.
        element_types: List of element types.
        ext: Extension of the created files.

    Note:
        A file is created for each element types.
    """
    for filename in template_filenames:

        for type_name in element_types:

            context = {
                'type' : type_name,
                'element_list' : ELEMENT_TYPES
            }

            context.update(template_context)

            generate_template_files(logger, template_filenames, template_environment, context, '_%s%s' % (type_name, ext))


def generate_following_type_and_index(logger, template_filenames, template_environment, template_context, element_types, index_types, ext, tabu_combination=None):
    """
    Generate Cython code.

    Args:
        logger: logging engine.
        template_filenames: List of file names.
        environment: An Jinja2 Environment.
        element_types: List of element types for containers.
        index_types: List of types.
        ext: Extension of the created files.
        tabu_combination: dict with tabu combinations of element and index types, like so:

            tabu_combination['type_X']['index_Y'] = False/True

            First element **must** be an element type and second element **must** be an index type.

    Note:
        A file is created for each element **and** index types.
    """
    context = {
        'type_list': ELEMENT_TYPES,
        'index_list': INDEX_TYPES
    }

    for filename in template_filenames:

        for __type_name in element_types:
            for index_name in index_types:
                if tabu_combination is not None:
                    try:
                        if tabu_combination[__type_name][index_name]:
                            continue
                    except:
                        pass

                # update context
                context['type'] = __type_name
                context['index'] = index_name

                context.update(template_context)

                generate_template_files(logger, template_filenames, template_environment, context, '_%s_%s%s' % (index_name, __type_name, ext))

#################################################################################################
# SETUP FILE
#################################################################################################
SETUP_FILE = os.path.join(PATH, 'setup.cpy')
SETUP_PY_FILE = os.path.join(PATH, 'setup.py')

#################################################################################################
# GENERIC TYPES
#################################################################################################
TYPES_DIR = os.path.join(PATH, 'cysparse', 'types')

TYPES_GENERIC_TYPES_DECLARATION_FILE = os.path.join(TYPES_DIR, 'cysparse_generic_types.cpd')
TYPES_GENERIC_TYPES_DEFINITION_FILE = os.path.join(TYPES_DIR, 'cysparse_generic_types.cpx')

#################################################################################################
# SPARSE
#################################################################################################
SPARSE_DIR = os.path.join(PATH, 'cysparse', 'sparse')

### sparse_utils
SPARSE_SPARSE_UTILS_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 'sparse_utils', 'generic')

SPARSE_SPARSE_UTILS_GENERATE_INDICES_DECLARATION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'generate_indices.cpd')]
SPARSE_SPARSE_UTILS_GENERATE_INDICES_DEFINITION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'generate_indices.cpx')]

SPARSE_SPARSE_UTILS_FIND_DECLARATION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'find.cpd')]
SPARSE_SPARSE_UTILS_FIND_DEFINITION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'find.cpx')]

SPARSE_SPARSE_UTILS_MATRIX_TRANSLATIONS_DECLARATION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'matrix_translations.cpd')]
SPARSE_SPARSE_UTILS_MATRIX_TRANSLATIONS_DEFINITION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'matrix_translations.cpx')]


SPARSE_SPARSE_UTILS_PRINT_DECLARATION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'print.cpd')]
SPARSE_SPARSE_UTILS_PRINT_DEFINITION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'print.cpx')]

SPARSE_SPARSE_UTILS_SORT_INDICES_DECLARATION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'sort_indices.cpd')]
SPARSE_SPARSE_UTILS_SORT_INDICES_DEFINITION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'sort_indices.cpx')]


#################################################################################################
# SPARSE MATRICES
#################################################################################################

##########################################
### SparseMatrix
##########################################
SPARSE_MATRIX_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 's_mat_matrices')

SPARSE_MATRIX_INCLUDE_FILES = glob.glob(os.path.join(SPARSE_MATRIX_TEMPLATE_DIR, '*.cpi'))
SPARSE_MATRIX_DECLARATION_FILES = glob.glob(os.path.join(SPARSE_MATRIX_TEMPLATE_DIR, '*.cpd'))
SPARSE_MATRIX_DEFINITION_FILES = glob.glob(os.path.join(SPARSE_MATRIX_TEMPLATE_DIR, '*.cpx'))

##########################################
### TransposedSparseMatrix
##########################################
SPARSE_MATRIX_PROXIES_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 'sparse_proxies')

SPARSE_MATRIX_PROXIES_TRANSPOSE_DECLARATION_FILE = os.path.join(SPARSE_MATRIX_PROXIES_TEMPLATE_DIR, 't_mat.cpd')
SPARSE_MATRIX_PROXIES_TRANSPOSE_DEFINITION_FILE = os.path.join(SPARSE_MATRIX_PROXIES_TEMPLATE_DIR, 't_mat.cpx')

##########################################
### ConjugateTransposedSparseMatrix
##########################################
SPARSE_MATRIX_PROXIES_GENERIC_TEMPLATE_DIR = os.path.join(SPARSE_MATRIX_PROXIES_TEMPLATE_DIR, 'complex_generic')

SPARSE_MATRIX_PROXIES_GENERIC_DECLARATION_FILES = glob.glob(os.path.join(SPARSE_MATRIX_PROXIES_GENERIC_TEMPLATE_DIR, '*.cpd'))
SPARSE_MATRIX_PROXIES_GENERIC_DEFINITION_FILES = glob.glob(os.path.join(SPARSE_MATRIX_PROXIES_GENERIC_TEMPLATE_DIR, '*.cpx'))

##########################################
### LLSparseMatrix
##########################################
LL_SPARSE_MATRIX_BASE_FILE = os.path.join(SPARSE_DIR, 'll_mat.cpx')

LL_SPARSE_MATRIX_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 'll_mat_matrices')

LL_SPARSE_MATRIX_INCLUDE_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_TEMPLATE_DIR, '*.cpi'))
LL_SPARSE_MATRIX_DECLARATION_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_TEMPLATE_DIR, '*.cpd'))
LL_SPARSE_MATRIX_DEFINITION_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_TEMPLATE_DIR, '*.cpx'))

### LLSparseMatrix kernel
LL_SPARSE_MATRIX_KERNEL_TEMPLATE_DIR = os.path.join(LL_SPARSE_MATRIX_TEMPLATE_DIR, 'll_mat_kernel')
LL_SPARSE_MATRIX_KERNEL_INCLUDE_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_KERNEL_TEMPLATE_DIR, '*.cpi'))

### LLSparseMatrix helpers
LL_SPARSE_MATRIX_HELPERS_TEMPLATE_DIR = os.path.join(LL_SPARSE_MATRIX_TEMPLATE_DIR, 'll_mat_helpers')
LL_SPARSE_MATRIX_HELPERS_INCLUDE_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_HELPERS_TEMPLATE_DIR, '*.cpi'))

### LLSparseMatrix IO
LL_SPARSE_MATRIX_IO_TEMPLATE_DIR = os.path.join(LL_SPARSE_MATRIX_TEMPLATE_DIR, 'll_mat_IO')

LL_SPARSE_MATRIX_MM_IO_INCLUDE_FILE = os.path.join(LL_SPARSE_MATRIX_IO_TEMPLATE_DIR, 'll_mat_mm.cpi')

### LLSparseMatrix construcors
LL_SPARSE_MATRIX_CONSTRUCTORS_TEMPLATE_DIR = os.path.join(LL_SPARSE_MATRIX_TEMPLATE_DIR, 'll_mat_constructors')
LL_SPARSE_MATRIX_CONSTRUCTORS_INCLUDE_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_CONSTRUCTORS_TEMPLATE_DIR, '*.cpi'))

##########################################
### LLSparseMatrixView
##########################################
LL_SPARSE_MATRIX_VIEW_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 'll_mat_views')

LL_SPARSE_MATRIX_VIEW_INCLUDE_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_VIEW_TEMPLATE_DIR, '*.cpi'))
LL_SPARSE_MATRIX_VIEW_DECLARATION_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_VIEW_TEMPLATE_DIR, '*.cpd'))
LL_SPARSE_MATRIX_VIEW_DEFINITION_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_VIEW_TEMPLATE_DIR, '*.cpx'))

##########################################
### CSRSparseMatrix
##########################################
CSR_SPARSE_MATRIX_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 'csr_mat_matrices')
CSR_SPARSE_MATRIX_DECLARATION_FILES = glob.glob(os.path.join(CSR_SPARSE_MATRIX_TEMPLATE_DIR, '*.cpd'))
CSR_SPARSE_MATRIX_DEFINITION_FILES = glob.glob(os.path.join(CSR_SPARSE_MATRIX_TEMPLATE_DIR, '*.cpx'))

### CSRSparseMatrix kernel
CSR_SPARSE_MATRIX_KERNEL_TEMPLATE_DIR = os.path.join(CSR_SPARSE_MATRIX_TEMPLATE_DIR, 'csr_mat_kernel')
CSR_SPARSE_MATRIX_KERNEL_INCLUDE_FILES = glob.glob(os.path.join(CSR_SPARSE_MATRIX_KERNEL_TEMPLATE_DIR, '*.cpi'))

### CSRSparseMatrix helpers
CSR_SPARSE_MATRIX_HELPERS_TEMPLATE_DIR = os.path.join(CSR_SPARSE_MATRIX_TEMPLATE_DIR, 'csr_mat_helpers')
CSR_SPARSE_MATRIX_HELPERS_INCLUDE_FILES = glob.glob(os.path.join(CSR_SPARSE_MATRIX_HELPERS_TEMPLATE_DIR, '*.cpi'))

##########################################
### CSCSparseMatrix
##########################################
CSC_SPARSE_MATRIX_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 'csc_mat_matrices')
CSC_SPARSE_MATRIX_DECLARATION_FILES = glob.glob(os.path.join(CSC_SPARSE_MATRIX_TEMPLATE_DIR, '*.cpd'))
CSC_SPARSE_MATRIX_DEFINITION_FILES = glob.glob(os.path.join(CSC_SPARSE_MATRIX_TEMPLATE_DIR, '*.cpx'))

### CSRSparseMatrix kernel
CSC_SPARSE_MATRIX_KERNEL_TEMPLATE_DIR = os.path.join(CSC_SPARSE_MATRIX_TEMPLATE_DIR, 'csc_mat_kernel')
CSC_SPARSE_MATRIX_KERNEL_INCLUDE_FILES = glob.glob(os.path.join(CSC_SPARSE_MATRIX_KERNEL_TEMPLATE_DIR, '*.cpi'))

### CSCSparseMatrix helpers
CSC_SPARSE_MATRIX_HELPERS_TEMPLATE_DIR = os.path.join(CSC_SPARSE_MATRIX_TEMPLATE_DIR, 'csc_mat_helpers')
CSC_SPARSE_MATRIX_HELPERS_INCLUDE_FILES = glob.glob(os.path.join(CSC_SPARSE_MATRIX_HELPERS_TEMPLATE_DIR, '*.cpi'))

##########################################
### CSBSparseMatrix
##########################################

#################################################################################################
# LINALG
#################################################################################################
LINALG_TEMPLATE_DIR = os.path.join(PATH, 'cysparse', 'linalg')

##########################################
### SuiteSparse
##########################################
LINALG_SUITESPARSE_TEMPLATE_DIR = os.path.join(LINALG_TEMPLATE_DIR, 'suitesparse')

# UMFPACK
LINALG_SUITESPARSE_UMFPACK_TEMPLATE_DIR = os.path.join(LINALG_SUITESPARSE_TEMPLATE_DIR, 'umfpack')

LINALG_SUITESPARSE_UMFPACK_DECLARATION_FILES = glob.glob(os.path.join(LINALG_SUITESPARSE_UMFPACK_TEMPLATE_DIR, '*.cpd'))
LINALG_SUITESPARSE_UMFPACK_DEFINITION_FILES = glob.glob(os.path.join(LINALG_SUITESPARSE_UMFPACK_TEMPLATE_DIR, '*.cpx'))

# CHOLMOD
LINALG_SUITESPARSE_CHOLMOD_TEMPLATE_DIR = os.path.join(LINALG_SUITESPARSE_TEMPLATE_DIR, 'cholmod')

LINALG_SUITESPARSE_CHOLMOD_DECLARATION_FILES = glob.glob(os.path.join(LINALG_SUITESPARSE_CHOLMOD_TEMPLATE_DIR, '*.cpd'))
LINALG_SUITESPARSE_CHOLMOD_DEFINITION_FILES = glob.glob(os.path.join(LINALG_SUITESPARSE_CHOLMOD_TEMPLATE_DIR, '*.cpx'))

##########################################
### MUMPS
##########################################
LINALG_MUMPS_TEMPLATE_DIR = os.path.join(LINALG_TEMPLATE_DIR, 'mumps')

LINALG_MUMPS_FACTORY_METHOD_FILE = os.path.join(LINALG_TEMPLATE_DIR, 'mumps_context.cpy')

LINALG_MUMPS_DECLARATION_FILES = glob.glob(os.path.join(LINALG_MUMPS_TEMPLATE_DIR, '*.cpd'))
LINALG_MUMPS_DEFINITION_FILES = glob.glob(os.path.join(LINALG_MUMPS_TEMPLATE_DIR, '*.cpx'))

#################################################################################################
# TESTS
#################################################################################################
TESTS_TEMPLATE_DIR = os.path.join(PATH, 'tests')

# SPARSE MATRICES
TESTS_CSC_SPARSE_MATRIX_GENERIC_TEST_DIR = os.path.join(TESTS_TEMPLATE_DIR, 'cysparse', 'sparse', 'csc_mat_matrices', 'generic')
TESTS_CSC_SPARSE_MATRIX_GENERIC_TEST_FILES = glob.glob(os.path.join(TESTS_CSC_SPARSE_MATRIX_GENERIC_TEST_DIR, '*.cpy'))

TESTS_CSR_SPARSE_MATRIX_GENERIC_TEST_DIR = os.path.join(TESTS_TEMPLATE_DIR, 'cysparse', 'sparse', 'csr_mat_matrices', 'generic')
TESTS_CSR_SPARSE_MATRIX_GENERIC_TEST_FILES = glob.glob(os.path.join(TESTS_CSR_SPARSE_MATRIX_GENERIC_TEST_DIR, '*.cpy'))

TESTS_LL_SPARSE_MATRIX_VIEW_GENERIC_TEST_DIR = os.path.join(TESTS_TEMPLATE_DIR, 'cysparse', 'sparse', 'll_mat_views', 'generic')
TESTS_LL_SPARSE_MATRIX_VIEW_GENERIC_TEST_FILES = glob.glob(os.path.join(TESTS_LL_SPARSE_MATRIX_VIEW_GENERIC_TEST_DIR, '*.cpy'))

TESTS_SPARSE_MATRIX_COMMON_OPERATIONS_GENERIC_TEST_DIR = os.path.join(TESTS_TEMPLATE_DIR, 'cysparse', 'sparse', 'common_operations', 'generic')
TESTS_SPARSE_MATRIX_COMMON_OPERATIONS_GENERIC_TEST_FILES = glob.glob(os.path.join(TESTS_SPARSE_MATRIX_COMMON_OPERATIONS_GENERIC_TEST_DIR, '*.cpy'))

# LINALG
TESTS_LINALG_DIR = os.path.join(TESTS_TEMPLATE_DIR, 'cysparse', 'linalg')

# SUITESPARSE
TESTS_SUITESPARSE_DIR = os.path.join(TESTS_LINALG_DIR, 'suitesparse')
# Umfpack
TESTS_UMFPACK_GENERIC_DIR = os.path.join(TESTS_SUITESPARSE_DIR, 'umfpack', 'generic')
TESTS_UMFPACK_GENERIC_TEST_FILES = glob.glob(os.path.join(TESTS_UMFPACK_GENERIC_DIR, '*.cpy'))



#################################################################################################
# MAIN
#################################################################################################
if __name__ == "__main__":

    # line arguments
    parser = make_parser()
    arg_options = parser.parse_args()

    # read cysparse.cfg
    cysparse_config = ConfigParser.SafeConfigParser()
    cysparse_config.read('cysparse.cfg')

    #######################################
    # CONDITIONAL CODE GENERATION
    #######################################
    # MUMPS
    # test if compiled lib has been compiled in 64 or 32 bits
    MUMPS_INT = None
    if cysparse_config.getboolean('MUMPS', 'mumps_compiled_in_64bits'):
        MUMPS_INT = 'INT64_t'
    else:
        MUMPS_INT = 'INT32_t'

    MUMPS_INDEX_TYPES = [MUMPS_INT]

    #######################################
    # END CONDITIONAL CODE GENERATION
    #######################################

    #################################################################################################
    # COMMON STUFF
    #################################################################################################
    # TODO: grab this from cysparse_types.pxd or at least from a one common file
    BASIC_TYPES = ['INT32_t', 'UINT32_t', 'INT64_t', 'UINT64_t', 'FLOAT32_t', 'FLOAT64_t', 'FLOAT128_t', 'COMPLEX64_t', 'COMPLEX128_t', 'COMPLEX256_t']
    ELEMENT_TYPES = ['INT32_t', 'INT64_t', 'FLOAT32_t', 'FLOAT64_t', 'FLOAT128_t', 'COMPLEX64_t', 'COMPLEX128_t', 'COMPLEX256_t']
    INDEX_TYPES = ['INT32_t', 'INT64_t']
    INTEGER_ELEMENT_TYPES = ['INT32_t', 'INT64_t']
    REAL_ELEMENT_TYPES = ['FLOAT32_t', 'FLOAT64_t', 'FLOAT128_t']
    COMPLEX_ELEMENT_TYPES = ['COMPLEX64_t', 'COMPLEX128_t', 'COMPLEX256_t']

    # Matrix market types
    MM_INDEX_TYPES = ['INT32_t', 'INT64_t']
    MM_ELEMENT_TYPES = ['INT64_t', 'FLOAT64_t', 'COMPLEX128_t']

    # Contexts
    # SuiteSparse
    # Umfpack
    UMFPACK_INDEX_TYPES = ['INT32_t', 'INT64_t']
    UMFPACK_ELEMENT_TYPES = ['FLOAT64_t', 'COMPLEX128_t']
    # Cholmod
    CHOLMOD_INDEX_TYPES = ['INT32_t', 'INT64_t']
    CHOLMOD_ELEMENT_TYPES = ['FLOAT64_t', 'COMPLEX128_t']

    # MUMPS
    # This list is defined above in the conditional part
    # MUMPS_INDEX_TYPES = [MUMPS_INT]
    #MUMPS_ELEMENT_TYPES = ['FLOAT32_t', 'FLOAT64_t', 'COMPLEX64_t', 'COMPLEX128_t']
    MUMPS_ELEMENT_TYPES = ['FLOAT32_t', 'FLOAT64_t']

    # when coding
    #ELEMENT_TYPES = ['FLOAT64_t']
    #ELEMENT_TYPES = ['COMPLEX64_t']
    #UMFPACK_INDEX_TYPES = ['INT32_t']
    #UMFPACK_ELEMENT_TYPES = ['FLOAT64_t']

    GENERAL_CONTEXT = {
                        'basic_type_list' : BASIC_TYPES,
                        'type_list': ELEMENT_TYPES,
                        'index_list' : INDEX_TYPES,
                        'integer_list' : INTEGER_ELEMENT_TYPES,
                        'real_list' : REAL_ELEMENT_TYPES,
                        'complex_list' : COMPLEX_ELEMENT_TYPES,
                        'mm_index_list' : MM_INDEX_TYPES,
                        'mm_type_list' : MM_ELEMENT_TYPES,
                        'umfpack_index_list': UMFPACK_INDEX_TYPES,
                        'umfpack_type_list' : UMFPACK_ELEMENT_TYPES,
                        'cholmod_index_list': CHOLMOD_INDEX_TYPES,
                        'cholmod_type_list': CHOLMOD_ELEMENT_TYPES,
                        'mumps_index_list': MUMPS_INDEX_TYPES,
                        'mumps_type_list': MUMPS_ELEMENT_TYPES,

                    }

    GENERAL_ENVIRONMENT = Environment(
        autoescape=False,
        loader=FileSystemLoader('/'), # we use absolute filenames
        trim_blocks=False,
        variable_start_string='@',
        variable_end_string='@')

    GENERAL_ENVIRONMENT.filters['type2enum'] = type2enum
    GENERAL_ENVIRONMENT.filters['cysparse_type_to_numpy_c_type'] = cysparse_type_to_numpy_c_type
    GENERAL_ENVIRONMENT.filters['cysparse_type_to_numpy_type'] = cysparse_type_to_numpy_type
    GENERAL_ENVIRONMENT.filters['cysparse_type_to_real_sum_cysparse_type'] = cysparse_type_to_real_sum_cysparse_type
    GENERAL_ENVIRONMENT.filters['cysparse_type_to_numpy_enum_type'] = cysparse_type_to_numpy_enum_type
    GENERAL_ENVIRONMENT.filters['cysparse_real_type_from_real_cysparse_complex_type'] = cysparse_real_type_from_real_cysparse_complex_type
    GENERAL_ENVIRONMENT.filters['cysparse_real_type_to_umfpack_family'] = cysparse_real_type_to_umfpack_family
    GENERAL_ENVIRONMENT.filters['cysparse_real_type_to_mumps_family'] = cysparse_real_type_to_mumps_family

    #################################################################################################
    # END COMMON STUFF
    #################################################################################################

    # create logger
    logger_name = cysparse_config.get('CODE_GENERATION', 'log_name')
    if logger_name == '':
        logger_name = 'cysparse_generate_code'

    logger = logging.getLogger(logger_name)

    # levels
    log_level = LOG_LEVELS[cysparse_config.get('CODE_GENERATION', 'log_level')]
    console_log_level = LOG_LEVELS[cysparse_config.get('CODE_GENERATION', 'console_log_level')]
    file_log_level = LOG_LEVELS[cysparse_config.get('CODE_GENERATION', 'file_log_level')]

    logger.setLevel(log_level)

    # create console handler and set logging level
    ch = logging.StreamHandler()
    ch.setLevel(console_log_level)

    # create file handler and set logging level
    log_file_name = logger_name + '.log'
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(file_log_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info('*' * 100)
    logger.info('*' * 100)
    logger.info("Start some action(s)")

    action = False

    if arg_options.setup or arg_options.all:
        action = True
        logger.info("Act for setup file")

        if arg_options.clean:
            clean_cython_files(logger, PATH, [SETUP_PY_FILE])
        else:
            generate_template_files(logger, [SETUP_FILE], GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.py')

    if arg_options.generic_types or arg_options.all:
        action = True
        logger.info("Act for generic types")

        if arg_options.clean:
            clean_cython_files(logger, SPARSE_DIR, [TYPES_GENERIC_TYPES_DECLARATION_FILE[:-4] + '.pxd'])
            clean_cython_files(logger, SPARSE_DIR, [TYPES_GENERIC_TYPES_DEFINITION_FILE[:-4] + '.pyx'])

        else:
            ###############################
            # Types
            ###############################
            # generic types
            generate_template_files(logger, [TYPES_GENERIC_TYPES_DECLARATION_FILE], GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.pxd')
            generate_template_files(logger, [TYPES_GENERIC_TYPES_DEFINITION_FILE], GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.pyx')

    if arg_options.matrices or arg_options.all:
        action = True
        logger.info("Act for matrices")

        if arg_options.clean:
            logger.info("Clean generated files")
            # Sparse
            clean_cython_files(logger, SPARSE_SPARSE_UTILS_TEMPLATE_DIR)


            # SparseMatrix
            clean_cython_files(logger, SPARSE_MATRIX_TEMPLATE_DIR)

            # TransposedSparseMatrix
            clean_cython_files(logger, SPARSE_MATRIX_PROXIES_TEMPLATE_DIR, [SPARSE_MATRIX_PROXIES_TRANSPOSE_DECLARATION_FILE[:-4] + '.pxd'])
            clean_cython_files(logger, SPARSE_MATRIX_PROXIES_TEMPLATE_DIR, [SPARSE_MATRIX_PROXIES_TRANSPOSE_DEFINITION_FILE[:-4] + '.pyx'])

            # ConjugateTransposedSparseMatrix
            clean_cython_files(logger, SPARSE_MATRIX_PROXIES_GENERIC_TEMPLATE_DIR)

            # LLSparseMatrix
            clean_cython_files(logger, LL_SPARSE_MATRIX_TEMPLATE_DIR)
            clean_cython_files(logger, SPARSE_DIR, [LL_SPARSE_MATRIX_BASE_FILE[:-4] + '.pyx'])
            # LLSparseMatrix kernel
            clean_cython_files(logger, LL_SPARSE_MATRIX_KERNEL_TEMPLATE_DIR)
            # LLSparseMatrix helpers
            clean_cython_files(logger, LL_SPARSE_MATRIX_HELPERS_TEMPLATE_DIR)

            # LLSparseMatrix IO
            clean_cython_files(logger, LL_SPARSE_MATRIX_IO_TEMPLATE_DIR)
            # LLSparseMatrix constructors
            clean_cython_files(logger, LL_SPARSE_MATRIX_CONSTRUCTORS_TEMPLATE_DIR)


            # LLSparseMatrixView
            clean_cython_files(logger, LL_SPARSE_MATRIX_VIEW_TEMPLATE_DIR)


            # CSRSparseMatrix
            clean_cython_files(logger, CSR_SPARSE_MATRIX_TEMPLATE_DIR)
            # CSRSparseMatrix kernel
            clean_cython_files(logger, CSR_SPARSE_MATRIX_KERNEL_TEMPLATE_DIR)
            # CSRSparseMatrix helpers
            clean_cython_files(logger, CSR_SPARSE_MATRIX_HELPERS_TEMPLATE_DIR)


            # CSCSparseMatrix
            clean_cython_files(logger, CSC_SPARSE_MATRIX_TEMPLATE_DIR)
            # CSCSparseMatrix kernel
            clean_cython_files(logger, CSC_SPARSE_MATRIX_KERNEL_TEMPLATE_DIR)
            # CSCSparseMatrix helpers
            clean_cython_files(logger, CSC_SPARSE_MATRIX_HELPERS_TEMPLATE_DIR)

        else:
            logger.info("Generate code files")

            ###############################
            # Sparse
            ###############################
            ### include_utils
            # generate_indices_@index@.pxd and generate_indices_@index@.pyx
            generate_following_only_index(logger, SPARSE_SPARSE_UTILS_GENERATE_INDICES_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, INDEX_TYPES, '.pxd')
            generate_following_only_index(logger, SPARSE_SPARSE_UTILS_GENERATE_INDICES_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, INDEX_TYPES, '.pyx')

            # sort_indices_@index@.pxd and sort_indices_@index@.pyx
            generate_following_only_index(logger, SPARSE_SPARSE_UTILS_SORT_INDICES_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, INDEX_TYPES, '.pxd')
            generate_following_only_index(logger, SPARSE_SPARSE_UTILS_SORT_INDICES_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, INDEX_TYPES, '.pyx')

            # element_to_string_@type@
            generate_following_only_type(logger, SPARSE_SPARSE_UTILS_PRINT_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, '.pxd')
            generate_following_only_type(logger, SPARSE_SPARSE_UTILS_PRINT_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, '.pyx')

            # find_@index@_@type@.pxd and find_@index@_@type@.pyx
            generate_following_type_and_index(logger, SPARSE_SPARSE_UTILS_FIND_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, SPARSE_SPARSE_UTILS_FIND_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pyx')

            # matrix_translations_@index@_@type@.pxd and matrix_translations_@index@_@type@.pyx
            generate_following_type_and_index(logger, SPARSE_SPARSE_UTILS_MATRIX_TRANSLATIONS_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, SPARSE_SPARSE_UTILS_MATRIX_TRANSLATIONS_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pyx')

            ###############################
            # SparseMatrix
            ###############################
            # sparse_matrix_@index@_@type@.pxd and sparse_matrix_@index@_@type@.pyx
            generate_following_type_and_index(logger, SPARSE_MATRIX_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, SPARSE_MATRIX_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pyx')

            # TODO: add the possibility to use tabu in for loops in Jinja2 (i.e. inside a template file)....
            #tabu = {}
            #tabu['INT32_t'] = {}
            #tabu['INT32_t']['INT64_t'] = True

            ###############################
            # Sparse proxies
            ###############################
            # Transpose
            # t_mat.pxd and t_mat.pyx
            generate_template_files(logger, [SPARSE_MATRIX_PROXIES_TRANSPOSE_DECLARATION_FILE], GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.pxd')
            generate_template_files(logger, [SPARSE_MATRIX_PROXIES_TRANSPOSE_DEFINITION_FILE], GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.pyx')

            # complex generic
            generate_following_type_and_index(logger, SPARSE_MATRIX_PROXIES_GENERIC_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, COMPLEX_ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, SPARSE_MATRIX_PROXIES_GENERIC_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, COMPLEX_ELEMENT_TYPES, INDEX_TYPES, '.pyx')

            ###############################
            # LLSparseMatrix
            ###############################
            # ll_mat.pyx
            generate_template_files(logger, [LL_SPARSE_MATRIX_BASE_FILE], GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.pyx')

            # ll_mat_@index@_@type@.pxd and ll_mat_@index@_@type@.pyx
            generate_following_type_and_index(logger, LL_SPARSE_MATRIX_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, LL_SPARSE_MATRIX_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pyx')

            # kernel
            # ll_mat_assignment_kernel.cpi
            generate_following_type_and_index(logger, LL_SPARSE_MATRIX_KERNEL_INCLUDE_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxi')

            # helpers
            generate_following_type_and_index(logger, LL_SPARSE_MATRIX_HELPERS_INCLUDE_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxi')

            # LLSparseMatrix IO
            # Matrix Market
            generate_following_type_and_index(logger, [LL_SPARSE_MATRIX_MM_IO_INCLUDE_FILE], GENERAL_ENVIRONMENT, GENERAL_CONTEXT, MM_ELEMENT_TYPES, MM_INDEX_TYPES, '.pxi')

            # LLSparseMatrix Constructors
            generate_following_type_and_index(logger, LL_SPARSE_MATRIX_CONSTRUCTORS_INCLUDE_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxi')

            ###############################
            # LLSparseMatrixView
            ###############################
            # ll_mat_view_@index@_@type@.pxd and ll_mat_view_@index@_@type@.pyx
            generate_following_type_and_index(logger, LL_SPARSE_MATRIX_VIEW_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, LL_SPARSE_MATRIX_VIEW_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pyx')

            ###############################
            # CSRSparseMatrix
            ###############################
            # csr_mat_@index@_@type@.pxd and csr_mat_@index@_@type@.pyx
            generate_following_type_and_index(logger, CSR_SPARSE_MATRIX_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, CSR_SPARSE_MATRIX_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pyx')

            # kernel
            generate_following_type_and_index(logger, CSR_SPARSE_MATRIX_KERNEL_INCLUDE_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxi')

            # helpers
            generate_following_type_and_index(logger, CSR_SPARSE_MATRIX_HELPERS_INCLUDE_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxi')

            ###############################
            # CSCSparseMatrix
            ###############################
            # csc_mat_@index@_@type@.pxd and csc_mat_@index@_@type@.pyx
            generate_following_type_and_index(logger, CSC_SPARSE_MATRIX_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, CSC_SPARSE_MATRIX_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pyx')

            # kernel
            generate_following_type_and_index(logger, CSC_SPARSE_MATRIX_KERNEL_INCLUDE_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxi')

            # helpers
            generate_following_type_and_index(logger, CSC_SPARSE_MATRIX_HELPERS_INCLUDE_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxi')

    if arg_options.linalg or arg_options.all:
        action = True
        logger.info("Act for generic contexts")

        if arg_options.clean:
            # SuiteSparse
            # Umfpack
            clean_cython_files(logger, LINALG_SUITESPARSE_UMFPACK_TEMPLATE_DIR)
            # Cholmod
            clean_cython_files(logger, LINALG_SUITESPARSE_CHOLMOD_TEMPLATE_DIR)

            # MUMPS
            clean_cython_files(logger, TESTS_LINALG_DIR, [LINALG_MUMPS_FACTORY_METHOD_FILE[:-4] + '.py'])
            clean_cython_files(logger, LINALG_MUMPS_TEMPLATE_DIR)
            
        else:
            ###############################
            # SuiteSparse
            ###############################
            # Umfpack
            generate_following_type_and_index(logger, LINALG_SUITESPARSE_UMFPACK_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, UMFPACK_ELEMENT_TYPES, UMFPACK_INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, LINALG_SUITESPARSE_UMFPACK_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, UMFPACK_ELEMENT_TYPES, UMFPACK_INDEX_TYPES, '.pyx')
            # Cholmod
            generate_following_type_and_index(logger, LINALG_SUITESPARSE_CHOLMOD_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, CHOLMOD_ELEMENT_TYPES, CHOLMOD_INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, LINALG_SUITESPARSE_CHOLMOD_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, CHOLMOD_ELEMENT_TYPES, CHOLMOD_INDEX_TYPES, '.pyx')

            ###############################
            # MUMPS
            ###############################
            generate_template_files(logger, [LINALG_MUMPS_FACTORY_METHOD_FILE], GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.py')

            generate_following_type_and_index(logger, LINALG_MUMPS_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, MUMPS_ELEMENT_TYPES, MUMPS_INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, LINALG_MUMPS_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, MUMPS_ELEMENT_TYPES, MUMPS_INDEX_TYPES, '.pyx')

    if arg_options.tests or arg_options.all:
        action = True
        logger.info("Act for generic tests")

        if arg_options.clean:
            clean_cython_files(logger, TESTS_CSC_SPARSE_MATRIX_GENERIC_TEST_DIR, find_files(TESTS_CSC_SPARSE_MATRIX_GENERIC_TEST_DIR, '*.py', False, True))
            clean_cython_files(logger, TESTS_CSR_SPARSE_MATRIX_GENERIC_TEST_DIR, find_files(TESTS_CSR_SPARSE_MATRIX_GENERIC_TEST_DIR, '*.py', False, True))
            clean_cython_files(logger, TESTS_LL_SPARSE_MATRIX_VIEW_GENERIC_TEST_DIR, find_files(TESTS_LL_SPARSE_MATRIX_VIEW_GENERIC_TEST_DIR, '*.py', False, True))
            clean_cython_files(logger, TESTS_SPARSE_MATRIX_COMMON_OPERATIONS_GENERIC_TEST_DIR, find_files(TESTS_SPARSE_MATRIX_COMMON_OPERATIONS_GENERIC_TEST_DIR, '*.py', False, True))

            clean_cython_files(logger, TESTS_UMFPACK_GENERIC_DIR, find_files(TESTS_UMFPACK_GENERIC_DIR, '*.py', False, True))
        else:
            ###############################
            # Types
            ###############################
            # generic types tests
            # Sparse matrices
            generate_template_files(logger, TESTS_CSC_SPARSE_MATRIX_GENERIC_TEST_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.py')
            generate_template_files(logger, TESTS_CSR_SPARSE_MATRIX_GENERIC_TEST_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.py')
            generate_template_files(logger, TESTS_LL_SPARSE_MATRIX_VIEW_GENERIC_TEST_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.py')
            generate_template_files(logger, TESTS_SPARSE_MATRIX_COMMON_OPERATIONS_GENERIC_TEST_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.py')

            # linalg
            # SuiteSparse
            # Umfpack
            generate_template_files(logger, TESTS_UMFPACK_GENERIC_TEST_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.py')

    if not action:
        logger.warning("No action proceeded...")

    logger.info("Stop some action(s)")
