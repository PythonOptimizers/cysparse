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
import os
import sys
import glob

import argparse
import logging


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
    parser.add_argument("-l", "--log_level", help="Log level while processing actions.", required=False, default='INFO')
    parser.add_argument("-a", "--all", help="Create all action files.", action='store_true', required=False)

    parser.add_argument("-m", "--matrices", help="Create sparse matrices.", action='store_true', required=False)
    parser.add_argument("-s", "--setup", help="Create setup file.", action='store_true', required=False)
    parser.add_argument("-g", "--generic_types", help="Create generic types.", action='store_true', required=False)
    parser.add_argument("-t", "--tests", help="Create generic types.", action='store_true', required=False)
    parser.add_argument("-c", "--clean", help="Clean action files.", action='store_true', required=False)

    return parser


#################################################################################################
# JINJA2 FILTERS
#################################################################################################
def type2enum(type_name):
    enum_name = type_name[:-1]
    enum_name = enum_name + type_name[-1].upper()

    return enum_name

# OLD VERSION
# def cysparse_type_to_numpy_c_type(cysparse_type):
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


def cysparse_type_to_real_sum_cysparse_type(cysparse_type):
    """
    Returns the best *real* type for a **real** sum for a given type.

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
INDEX_MM_TYPES = ['INT32_t', 'INT64_t']
ELEMENT_MM_TYPES = ['INT64_t', 'FLOAT64_t', 'COMPLEX128_t']

# when coding
#ELEMENT_TYPES = ['FLOAT64_t']


GENERAL_CONTEXT = {
                    'basic_type_list' : BASIC_TYPES,
                    'type_list': ELEMENT_TYPES,
                    'index_list' : INDEX_TYPES,
                    'integer_list' : INTEGER_ELEMENT_TYPES,
                    'real_list' : REAL_ELEMENT_TYPES,
                    'complex_list' : COMPLEX_ELEMENT_TYPES,
                    'mm_index_list' : INDEX_MM_TYPES,
                    'mm_type_list' : ELEMENT_MM_TYPES
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


def clean_cython_files(logger, directory, file_list=None):
    """

    :param logger:
    :param directory:
    :param file_list: File name **must** be absolute!
    :return:
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
        index_types: List of types.
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
        'index_list' : INDEX_TYPES
    }

    for filename in template_filenames:

        for type_name in element_types:
            for index_name in index_types:
                if tabu_combination is not None:
                    try:
                        if tabu_combination[type_name][index_name]:
                            continue
                    except:
                        pass

                # update context
                context['type'] = type_name
                context['index'] = index_name

                context.update(template_context)

                generate_template_files(logger, template_filenames, template_environment, context, '_%s_%s%s' % (index_name, type_name, ext))

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
SPARSE_SPARSE_UTILS_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 'sparse_utils')

SPARSE_SPARSE_UTILS_GENERATE_INDICES_DECLARATION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'generate_indices.cpd')]
SPARSE_SPARSE_UTILS_GENERATE_INDICES_DEFINITION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'generate_indices.cpx')]

SPARSE_SPARSE_UTILS_FIND_DECLARATION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'find.cpd')]
SPARSE_SPARSE_UTILS_FIND_DEFINITION_FILES = [os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, 'find.cpx')]


#################################################################################################
# SPARSE MATRICES
#################################################################################################

### SparseMatrix

SPARSE_MATRIX_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 's_mat_matrices')

SPARSE_MATRIX_INCLUDE_FILES = glob.glob(os.path.join(SPARSE_MATRIX_TEMPLATE_DIR, '*.cpi'))
SPARSE_MATRIX_DECLARATION_FILES = glob.glob(os.path.join(SPARSE_MATRIX_TEMPLATE_DIR, '*.cpd'))
SPARSE_MATRIX_DEFINITION_FILES = glob.glob(os.path.join(SPARSE_MATRIX_TEMPLATE_DIR, '*.cpx'))


### LLSparseMatrix

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
LL_SPARSE_MATRIX_IO_INCLUDE_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_IO_TEMPLATE_DIR, '*.cpi'))


### CSRSparseMatrix

### CSCSparseMatrix

### CSBSparseMatrix

### LLSparseMatrixView
LL_SPARSE_MATRIX_VIEW_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 'll_mat_views')

LL_SPARSE_MATRIX_VIEW_INCLUDE_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_VIEW_TEMPLATE_DIR, '*.cpi'))
LL_SPARSE_MATRIX_VIEW_DECLARATION_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_VIEW_TEMPLATE_DIR, '*.cpd'))
LL_SPARSE_MATRIX_VIEW_DEFINITION_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_VIEW_TEMPLATE_DIR, '*.cpx'))

#################################################################################################
# MAIN
#################################################################################################
if __name__ == "__main__":

    # line arguments
    parser = make_parser()
    arg_options = parser.parse_args()

    # create logger
    logger_name = 'cysparse_generate_code'
    logger = logging.getLogger(logger_name)

    # levels
    log_level = LOG_LEVELS[arg_options.log_level]
    console_log_level = LOG_LEVELS['WARNING']       # CHANGE THIS
    file_log_level = LOG_LEVELS['INFO']             # CHANGE THIS

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
            # LLSparseMatrix
            clean_cython_files(logger, LL_SPARSE_MATRIX_TEMPLATE_DIR)
            clean_cython_files(logger, SPARSE_DIR, [LL_SPARSE_MATRIX_BASE_FILE[:-4] + '.pyx'])
            # LLSparseMatrix kernel
            clean_cython_files(logger, LL_SPARSE_MATRIX_KERNEL_TEMPLATE_DIR)
            # LLSparseMatrix helpers
            clean_cython_files(logger, LL_SPARSE_MATRIX_HELPERS_TEMPLATE_DIR)
            # LLSparseMatrix IO
            clean_cython_files(logger, LL_SPARSE_MATRIX_IO_TEMPLATE_DIR)
            # LLSparseMatrixView
            clean_cython_files(logger, LL_SPARSE_MATRIX_VIEW_TEMPLATE_DIR)

        else:
            logger.info("Generate code files")

            ###############################
            # Sparse
            ###############################
            ### include_utils
            # generate_indices_@index@.pxd and generate_indices_@index@.pyx
            generate_following_only_index(logger, SPARSE_SPARSE_UTILS_GENERATE_INDICES_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, INDEX_TYPES, '.pxd')
            generate_following_only_index(logger, SPARSE_SPARSE_UTILS_GENERATE_INDICES_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, INDEX_TYPES, '.pyx')

            generate_following_type_and_index(logger, SPARSE_SPARSE_UTILS_FIND_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, SPARSE_SPARSE_UTILS_FIND_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pyx')

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
            generate_following_type_and_index(logger, LL_SPARSE_MATRIX_IO_INCLUDE_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_MM_TYPES, INDEX_TYPES, '.pxi')

            ###############################
            # LLSparseMatrixView
            ###############################
            # ll_mat_view_@index@_@type@.pxd and ll_mat_view_@index@_@type@.pyx
            generate_following_type_and_index(logger, LL_SPARSE_MATRIX_VIEW_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, LL_SPARSE_MATRIX_VIEW_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pyx')

    if not action:
        logger.warning("No action proceeded...")

    logger.info("Stop some action(s)")
