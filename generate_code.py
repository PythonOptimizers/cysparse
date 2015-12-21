#!/usr/bin/env python
###############################################################################
# This script generates all templated code for CySparse
# It this the single one script to use before Cythonizing the CySparse library.
# This script is NOT automatically called by setup.py
#
# We use our internal library cygenja, using itself the Jinja2 template engine:
# http://jinja.pocoo.org/docs/dev/
#
###############################################################################

from cygenja.generator import Generator
from jinja2 import Environment, FileSystemLoader

import ConfigParser
import argparse
import os
import sys
import shutil
import logging

#####################################################
# PARSER
#####################################################
def make_parser():
    """
    Create a comment line argument parser.

    Returns:
        The command line parser.
    """
    parser = argparse.ArgumentParser(description='%s: a Cython code generator for the CySparse library' % os.path.basename(sys.argv[0]))
    parser.add_argument("-r", "--recursive", help="Act recursively", action='store_true', required=False)
    parser.add_argument("-c", "--clean", help="Clean generated files", action='store_true', required=False)
    parser.add_argument("-d", "--dry_run", help="Dry run: no action is taken", action='store_true', required=False)
    parser.add_argument("-f", "--force", help="Force generation no matter what", action='store_true', required=False)

    parser.add_argument('dir_pattern', nargs='?', default='.', help='Glob pattern')
    parser.add_argument('file_pattern', nargs='?', default='*.*', help='Fnmatch pattern')

    return parser

###################################################################s####################################################
# LOGGING
########################################################################################################################
LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }


def make_logger(cysparse_config):
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

    return logger

#######################################
# CONDITIONAL CODE GENERATION
#######################################
# type of platform? 32bits or 64bits?
is_64bits = sys.maxsize > 2**32
default_index_type_str = '32bits'
if is_64bits:
    default_index_type_str = '64bits'

# read cysparse.cfg
cysparse_config = ConfigParser.SafeConfigParser()
cysparse_config.read('cysparse.cfg')


# index type for LLSparseMatrix
DEFAULT_INDEX_TYPE = 'INT32_T'
if is_64bits:
    DEFAULT_INDEX_TYPE = 'INT64_T'

if cysparse_config.get('CODE_GENERATION', 'DEFAULT_INDEX_TYPE') == '32bits':
    DEFAULT_INDEX_TYPE = 'INT32_T'
elif cysparse_config.get('CODE_GENERATION', 'DEFAULT_INDEX_TYPE') == '64bits':
    DEFAULT_INDEX_TYPE = 'INT64_T'
else:
    # don't do anything: use platform's default
    pass

#####################################################
# COMMON STUFF
#####################################################
# TODO: grab this from common_types.pxd or at least from a one common file
BASIC_TYPES = ['INT32_t', 'UINT32_t', 'INT64_t', 'UINT64_t', 'FLOAT32_t', 'FLOAT64_t', 'FLOAT128_t', 'COMPLEX64_t', 'COMPLEX128_t', 'COMPLEX256_t']
ELEMENT_TYPES = ['INT32_t', 'INT64_t', 'FLOAT32_t', 'FLOAT64_t', 'FLOAT128_t', 'COMPLEX64_t', 'COMPLEX128_t', 'COMPLEX256_t']
INDEX_TYPES = ['INT32_t', 'INT64_t']
INTEGER_ELEMENT_TYPES = ['INT32_t', 'INT64_t']
REAL_ELEMENT_TYPES = ['FLOAT32_t', 'FLOAT64_t', 'FLOAT128_t']
COMPLEX_ELEMENT_TYPES = ['COMPLEX64_t', 'COMPLEX128_t', 'COMPLEX256_t']

# Matrix market types
MM_INDEX_TYPES = ['INT32_t', 'INT64_t']
MM_ELEMENT_TYPES = ['INT64_t', 'FLOAT64_t', 'COMPLEX128_t']

# when coding
#ELEMENT_TYPES = ['FLOAT64_t']
#ELEMENT_TYPES = ['COMPLEX64_t']
#ELEMENT_TYPES = ['COMPLEX256_t']

GENERAL_CONTEXT = {
                    'basic_type_list' : BASIC_TYPES,
                    'type_list': ELEMENT_TYPES,
                    'index_list' : INDEX_TYPES,
                    'default_index_type' : DEFAULT_INDEX_TYPE,
                    'integer_list' : INTEGER_ELEMENT_TYPES,
                    'real_list' : REAL_ELEMENT_TYPES,
                    'complex_list' : COMPLEX_ELEMENT_TYPES,
                    'mm_index_list' : MM_INDEX_TYPES,
                    'mm_type_list' : MM_ELEMENT_TYPES,
                  }

# For tests
MATRIX_CLASSES = {'LLSparseMatrix' : 'll_mat_matrices.ll_mat',
                  'CSCSparseMatrix' : 'csc_mat_matrices.csc_mat',
                  'CSRSparseMatrix' : 'csr_mat_matrices.csr_mat'}
MATRIX_VIEW_CLASSES = {'LLSparseMatrixView' : 'll_mat_views.ll_mat_views'}
MATRIX_PROXY_CLASSES = {'TransposedSparseMatrix' : 'sparse_proxies.t_mat',
                        'ConjugatedSparseMatrix' : 'sparse_proxies.complex_generic.conj_mat',
                       'ConjugateTransposedSparseMatrix' : 'sparse_proxies.complex_generic.h_mat'}

MATRIX_LIKE_CLASSES = {}
MATRIX_LIKE_CLASSES.update(MATRIX_CLASSES)
MATRIX_LIKE_CLASSES.update(MATRIX_VIEW_CLASSES)
MATRIX_LIKE_CLASSES.update(MATRIX_PROXY_CLASSES)


#####################################################
# ACTION FUNCTION
#####################################################
# GENERAL
def single_generation():
    """
    Only generate one file without any suffix.
    """
    yield '', GENERAL_CONTEXT


def generate_following_only_index():
    """
    Generate files following the index types.
    """
    GENERAL_CONTEXT['type'] = None
    for index in INDEX_TYPES:
        GENERAL_CONTEXT['index'] = index

        yield '_%s' % index, GENERAL_CONTEXT


def generate_following_only_element():
    """
    Generate files following the element types.
    """
    GENERAL_CONTEXT['index'] = None
    for type in ELEMENT_TYPES:
        GENERAL_CONTEXT['type'] = type

        yield '_%s' % type, GENERAL_CONTEXT


def generate_following_index_and_element():
    """
    Generate files following the index and element types.
    """
    for index in INDEX_TYPES:
        GENERAL_CONTEXT['index'] = index
        for type in ELEMENT_TYPES:
            GENERAL_CONTEXT['type'] = type
            yield '_%s_%s' % (index, type), GENERAL_CONTEXT


def generate_following_index_and_complex_element():
    """
    Generate files following the index and complex element types.
    """
    for index in INDEX_TYPES:
        GENERAL_CONTEXT['index'] = index
        for type in COMPLEX_ELEMENT_TYPES:
            GENERAL_CONTEXT['type'] = type
            yield '_%s_%s' % (index, type), GENERAL_CONTEXT


# Matrix Market
def generate_MM_following_index_and_element():
    """
    Generate files following the Matrix Market index and element types.
    """
    for index in MM_INDEX_TYPES:
        GENERAL_CONTEXT['index'] = index
        for type in MM_ELEMENT_TYPES:
            GENERAL_CONTEXT['type'] = type
            yield '_%s_%s' % (index, type), GENERAL_CONTEXT


# Tests
def generate_following_matrix_class_and_index_and_type():
    """
    Generate files following index, element and class types.

    This generator is for tests only.

    Warning:
        Class :class:`TransposedSparseMatrix` is not generated because of its special status.

    """
    for klass, directory in MATRIX_CLASSES.items():
        GENERAL_CONTEXT['class'] = klass
        GENERAL_CONTEXT['directory'] = directory
        for index in INDEX_TYPES:
            GENERAL_CONTEXT['index'] = index
            for type in ELEMENT_TYPES:
                GENERAL_CONTEXT['type'] = type
                yield '_%s_%s_%s' % (klass, index, type), GENERAL_CONTEXT


def generate_following_matrix_view_class_and_index_and_type():
    """
    Generate files following index, element and class types.

    This generator is for tests only.

    """
    for klass, directory in MATRIX_VIEW_CLASSES.items():
        GENERAL_CONTEXT['class'] = klass
        GENERAL_CONTEXT['directory'] = directory
        for index in INDEX_TYPES:
            GENERAL_CONTEXT['index'] = index
            for type in ELEMENT_TYPES:
                GENERAL_CONTEXT['type'] = type
                yield '_%s_%s_%s' % (klass, index, type), GENERAL_CONTEXT


def generate_following_complex_matrix_proxy_class_and_index_and_type():
    """
    Generate files following index, element and class types.

    This generator is for tests only.

    Warning:
        Class :class:`TransposedSparseMatrix` is not generated because of its special status.

    Note:
        We only take proxies for **complex** matrices!

    """
    for klass, directory in MATRIX_PROXY_CLASSES.items():
        if klass == 'TransposedSparseMatrix':
            continue
        GENERAL_CONTEXT['class'] = klass
        GENERAL_CONTEXT['directory'] = directory
        for index in INDEX_TYPES:
            GENERAL_CONTEXT['index'] = index
            for type in COMPLEX_ELEMENT_TYPES:
                GENERAL_CONTEXT['type'] = type
                yield '_%s_%s_%s' % (klass, index, type), GENERAL_CONTEXT


def generate_following_matrix_proxy_transposed_class_and_index_and_type():
    """
    Generate files following index, element and class types.

    This generator is for tests only.

    Warning:
        Only the class :class:`TransposedSparseMatrix` is generated.

    """
    for klass, directory in MATRIX_PROXY_CLASSES.items():
        if klass != 'TransposedSparseMatrix':
            continue
        GENERAL_CONTEXT['class'] = klass
        GENERAL_CONTEXT['directory'] = directory
        for index in INDEX_TYPES:
            GENERAL_CONTEXT['index'] = index
            for type in ELEMENT_TYPES:
                GENERAL_CONTEXT['type'] = type
                yield '_%s_%s_%s' % (klass, index, type), GENERAL_CONTEXT

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":

    ####################################################################################################################
    # init
    ####################################################################################################################
    # line arguments
    parser = make_parser()
    arg_options = parser.parse_args()

    # create logger
    logger = make_logger(cysparse_config=cysparse_config)

    # cygenja engine
    current_directory = os.path.dirname(os.path.abspath(__file__))
    jinja2_env = Environment(autoescape=False,
                            loader=FileSystemLoader('/'), # we use absolute filenames
                            trim_blocks=False,
                            variable_start_string='@',
                            variable_end_string='@')

    cygenja_engine = Generator(current_directory, jinja2_env, logger=logger)

    # register filters
    cygenja_engine.register_common_type_filters()

    # register extensions
    cygenja_engine.register_extension('.cpy', '.py')
    cygenja_engine.register_extension('.cpx', '.pyx')
    cygenja_engine.register_extension('.cpd', '.pxd')
    cygenja_engine.register_extension('.cpi', '.pxi')

    ####################################################################################################################
    # register actions
    ####################################################################################################################
    ########## Setup ############
    cygenja_engine.register_action('config', '*.*', single_generation)
    ########## TYPES ############
    cygenja_engine.register_action('cysparse/common_types', '*.*', single_generation)
    ########## Sparse ###########
    # CSC
    cygenja_engine.register_action('cysparse/sparse/csc_mat_matrices', '*.*', generate_following_index_and_element)
    cygenja_engine.register_action('cysparse/sparse/csc_mat_matrices/csc_mat_helpers', '*.cpi', generate_following_index_and_element)
    cygenja_engine.register_action('cysparse/sparse/csc_mat_matrices/csc_mat_kernel', '*.cpi', generate_following_index_and_element)
    # CSR
    cygenja_engine.register_action('cysparse/sparse/csr_mat_matrices', '*.*', generate_following_index_and_element)
    cygenja_engine.register_action('cysparse/sparse/csr_mat_matrices/csr_mat_helpers', '*.cpi', generate_following_index_and_element)
    cygenja_engine.register_action('cysparse/sparse/csr_mat_matrices/csr_mat_kernel', '*.cpi', generate_following_index_and_element)
    # LL
    cygenja_engine.register_action('cysparse/sparse/ll_mat_matrices', '*.*', generate_following_index_and_element)
    cygenja_engine.register_action('cysparse/sparse/ll_mat_matrices/ll_mat_IO', 'll_mat_mm.*', generate_MM_following_index_and_element)
    cygenja_engine.register_action('cysparse/sparse/ll_mat_matrices/ll_mat_constructors', '*.cpi', generate_following_index_and_element)
    cygenja_engine.register_action('cysparse/sparse/ll_mat_matrices/ll_mat_helpers', '*.cpi', generate_following_index_and_element)
    cygenja_engine.register_action('cysparse/sparse/ll_mat_matrices/ll_mat_kernel', '*.cpi', generate_following_index_and_element)
    # LL views
    cygenja_engine.register_action('cysparse/sparse/ll_mat_views', '*.*', generate_following_index_and_element)
    # S MAT
    cygenja_engine.register_action('cysparse/sparse/s_mat_matrices', '*.*', generate_following_index_and_element)
    # Proxies
    cygenja_engine.register_action('cysparse/sparse/sparse_proxies', '*.*', single_generation)
    cygenja_engine.register_action('cysparse/sparse/sparse_proxies/complex_generic', '*.*', generate_following_index_and_complex_element)
    # Sparse utils
    cygenja_engine.register_action('cysparse/sparse/sparse_utils/generic', 'find.*', generate_following_index_and_element)
    cygenja_engine.register_action('cysparse/sparse/sparse_utils/generic', 'generate_indices.*', generate_following_only_index)
    cygenja_engine.register_action('cysparse/sparse/sparse_utils/generic', 'matrix_translations.*', generate_following_index_and_element)
    cygenja_engine.register_action('cysparse/sparse/sparse_utils/generic', 'print.*', generate_following_only_element)
    cygenja_engine.register_action('cysparse/sparse/sparse_utils/generic', 'sort_indices.*', generate_following_only_index)
    # Sparse
    cygenja_engine.register_action('cysparse/sparse', '*.*', single_generation)

    # Tests
    cygenja_engine.register_action('tests/cysparse_/sparse/common_operations', 'test_common_attributes_matrices.*', generate_following_matrix_class_and_index_and_type)
    cygenja_engine.register_action('tests/cysparse_/sparse/common_operations', 'test_common_attributes_matrices_views.*', generate_following_matrix_view_class_and_index_and_type)
    cygenja_engine.register_action('tests/cysparse_/sparse/common_operations', 'test_common_attributes_matrices_proxies.*', generate_following_complex_matrix_proxy_class_and_index_and_type)
    cygenja_engine.register_action('tests/cysparse_/sparse/common_operations', 'test_common_attributes_matrices_proxies_transposed.*', generate_following_matrix_proxy_transposed_class_and_index_and_type)

    ####################################################################################################################
    # Generation
    ####################################################################################################################
    if arg_options.dry_run:
        cygenja_engine.generate(arg_options.dir_pattern, arg_options.file_pattern, action_ch='d', recursively=arg_options.recursive, force=arg_options.force)
    elif arg_options.clean:
        cygenja_engine.generate(arg_options.dir_pattern, arg_options.file_pattern, action_ch='c', recursively=arg_options.recursive, force=arg_options.force)
    else:
        cygenja_engine.generate(arg_options.dir_pattern, arg_options.file_pattern, action_ch='g', recursively=arg_options.recursive, force=arg_options.force)
        # special case for the setup.py file
        shutil.copy2(os.path.join('config', 'setup.py'), '.')
