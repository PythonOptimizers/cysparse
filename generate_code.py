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

from cysparse.utils.log_utils import make_logger

import ConfigParser
import argparse
import os
import sys
import shutil


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

# MUMPS
# test if compiled lib has been compiled in 64 or 32 bits
MUMPS_INT = None
if cysparse_config.getboolean('MUMPS', 'mumps_compiled_in_64bits'):
    MUMPS_INT = 'INT64_t'
else:
    MUMPS_INT = 'INT32_t'

MUMPS_INDEX_TYPES = [MUMPS_INT]

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

# SuiteSparse
# SPQR
SPQR_EXPERT_MODE = not cysparse_config.getboolean('SUITESPARSE', 'NEXPERT')

#####################################################
# COMMON STUFF
#####################################################
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
# SPQR
SPQR_INDEX_TYPES = ['INT32_t', 'INT64_t']
SPQR_ELEMENT_TYPES = ['FLOAT64_t', 'COMPLEX128_t']

# MUMPS
# This list is defined above in the conditional part
# MUMPS_INDEX_TYPES = [MUMPS_INT]
MUMPS_ELEMENT_TYPES = ['FLOAT32_t', 'FLOAT64_t', 'COMPLEX64_t', 'COMPLEX128_t']
#MUMPS_ELEMENT_TYPES = ['FLOAT32_t', 'FLOAT64_t']

# when coding
#ELEMENT_TYPES = ['FLOAT64_t']
#ELEMENT_TYPES = ['COMPLEX64_t']
#ELEMENT_TYPES = ['COMPLEX256_t']
#UMFPACK_INDEX_TYPES = ['INT32_t']
#UMFPACK_ELEMENT_TYPES = ['FLOAT64_t']

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
                    'umfpack_index_list': UMFPACK_INDEX_TYPES,
                    'umfpack_type_list' : UMFPACK_ELEMENT_TYPES,
                    'cholmod_index_list': CHOLMOD_INDEX_TYPES,
                    'cholmod_type_list': CHOLMOD_ELEMENT_TYPES,
                    'spqr_index_list': SPQR_INDEX_TYPES,
                    'spqr_type_list': SPQR_ELEMENT_TYPES,
                    'spqr_export_mode' : SPQR_EXPERT_MODE,
                    'mumps_index_list': MUMPS_INDEX_TYPES,
                    'mumps_type_list': MUMPS_ELEMENT_TYPES,

                }


#####################################################
# ACTION FUNCTION
#####################################################
# GENERAL
def single_generation():
    yield '', GENERAL_CONTEXT


def generate_following_only_index():
    """

    """
    for index in INDEX_TYPES:
        # keys 'index' and 'type' are redefined in **all** action function.
        GENERAL_CONTEXT['index'] = index
        GENERAL_CONTEXT['type'] = None
        yield '_%s' % index, GENERAL_CONTEXT


def generate_following_only_type():
    """

    """
    for type in ELEMENT_TYPES:
        # keys 'index' and 'type' are redefined in **all** action function.
        GENERAL_CONTEXT['index'] = None
        GENERAL_CONTEXT['type'] = type
        yield '_%s' % type, GENERAL_CONTEXT


def generate_following_index_and_type():
    """

    """
    for index in INDEX_TYPES:
        GENERAL_CONTEXT['index'] = index
        for type in ELEMENT_TYPES:
            GENERAL_CONTEXT['type'] = type
            yield '_%s_%s' % (index, type), GENERAL_CONTEXT


def generate_following_index_and_complex_type():
    """

    """
    for index in INDEX_TYPES:
        GENERAL_CONTEXT['index'] = index
        for type in COMPLEX_ELEMENT_TYPES:
            GENERAL_CONTEXT['type'] = type
            yield '_%s_%s' % (index, type), GENERAL_CONTEXT


# Matrix Market
def generate_MM_following_index_and_type():
    for index in MM_INDEX_TYPES:
        GENERAL_CONTEXT['index'] = index
        for type in MM_ELEMENT_TYPES:
            GENERAL_CONTEXT['type'] = type
            yield '_%s_%s' % (index, type), GENERAL_CONTEXT

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
    cygenja_engine = Generator(current_directory, logger=logger)

    print cygenja_engine.root_directory()

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
    cygenja_engine.register_action('cysparse/cysparse_types', '*.*', single_generation)
    ########## Sparse ###########
    # CSC
    cygenja_engine.register_action('cysparse/sparse/csc_mat_matrices', '*.*', generate_following_index_and_type)
    cygenja_engine.register_action('cysparse/sparse/csc_mat_matrices/csc_mat_helpers', '*.cpi', generate_following_index_and_type)
    cygenja_engine.register_action('cysparse/sparse/csc_mat_matrices/csc_mat_kernel', '*.cpi', generate_following_index_and_type)
    # CSR
    cygenja_engine.register_action('cysparse/sparse/csr_mat_matrices', '*.*', generate_following_index_and_type)
    cygenja_engine.register_action('cysparse/sparse/csr_mat_matrices/csr_mat_helpers', '*.cpi', generate_following_index_and_type)
    cygenja_engine.register_action('cysparse/sparse/csr_mat_matrices/csr_mat_kernel', '*.cpi', generate_following_index_and_type)
    # LL
    cygenja_engine.register_action('cysparse/sparse/ll_mat_matrices', '*.*', generate_following_index_and_type)
    cygenja_engine.register_action('cysparse/sparse/ll_mat_matrices/ll_mat_IO', 'll_mat_mm.*', generate_MM_following_index_and_type)
    cygenja_engine.register_action('cysparse/sparse/ll_mat_matrices/ll_mat_constructors', '*.cpi', generate_following_index_and_type)
    cygenja_engine.register_action('cysparse/sparse/ll_mat_matrices/ll_mat_helpers', '*.cpi', generate_following_index_and_type)
    cygenja_engine.register_action('cysparse/sparse/ll_mat_matrices/ll_mat_kernel', '*.cpi', generate_following_index_and_type)
    # LL views
    cygenja_engine.register_action('cysparse/sparse/ll_mat_views', '*.*', generate_following_index_and_type)
    # S MAT
    cygenja_engine.register_action('cysparse/sparse/s_mat_matrices', '*.*', generate_following_index_and_type)
    # Proxies
    cygenja_engine.register_action('cysparse/sparse/sparse_proxies', '*.*', single_generation)
    cygenja_engine.register_action('cysparse/sparse/sparse_proxies/complex_generic', '*.*', generate_following_index_and_complex_type)
    # Sparse utils
    cygenja_engine.register_action('cysparse/sparse/sparse_utils/generic', 'find.*', generate_following_index_and_type)
    cygenja_engine.register_action('cysparse/sparse/sparse_utils/generic', 'generate_indices.*', generate_following_only_index)
    cygenja_engine.register_action('cysparse/sparse/sparse_utils/generic', 'matrix_translations.*', generate_following_index_and_type)
    cygenja_engine.register_action('cysparse/sparse/sparse_utils/generic', 'print.*', generate_following_only_type)
    cygenja_engine.register_action('cysparse/sparse/sparse_utils/generic', 'sort_indices.*', generate_following_only_index)
    # Sparse
    cygenja_engine.register_action('cysparse/sparse', '*.*', single_generation)

    ####################################################################################################################
    # Generation
    ####################################################################################################################
    if arg_options.dry_run:
        cygenja_engine.generate(arg_options.dir_pattern, arg_options.file_pattern, action_ch='d', recursively=arg_options.recursive, force=arg_options.force)
    elif arg_options.clean:
        cygenja_engine.generate(arg_options.dir_pattern, arg_options.file_pattern, action_ch='c', recursively=arg_options.recursive, force=arg_options.force)
    else:
        cygenja_engine.generate(arg_options.dir_pattern, arg_options.file_pattern, action_ch='g', recursively=arg_options.recursive, force=arg_options.force)

    ####################################################################################################################
    # Special setup.py case
    ####################################################################################################################
    shutil.copy2(os.path.join('config', 'setup.py'), '.')