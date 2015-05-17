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
    #parser.add_argument('project_file', nargs='?', default=None, help="Project file: *.cdp or *.txt (import).")
    #parser.add_argument("-a", "--action", help="Batch version (no GUI). ACTION must be either '%s' or '%s' and "
    #                                           "requires a project file (*.cdp), like so: -a %s project_file.cdp." % (ACTION_DOCSTRING, ACTION_DOCUMENTATION, ACTION_DOCSTRING),  required=False)
    parser.add_argument("-m", "--matrices", help="Create sparse matrices.", action='store_true', required=False)
    parser.add_argument("-s", "--setup", help="Create setup file.", action='store_true', required=False)
    parser.add_argument("-c", "--clean", help="Clean action files.", action='store_true', required=False)
    #parser.add_argument("-w", "--wxptyhon_version", help="wxPython version to use. Use it at your own risk!", required=False, default=None)
    #parser.add_argument("-i", "--info", help="Returns some system info and exit.", action='store_true', required=False)

    return parser


#################################################################################################
# JINJA2 FILTERS
#################################################################################################
def type2enum(type_name):
    enum_name = type_name[:-1]
    enum_name = enum_name + type_name[-1].upper()

    return enum_name


#################################################################################################
# COMMON STUFF
#################################################################################################
# TODO: grab this from cysparse_types.pxd or at least from a one common file
ELEMENT_TYPES = ['INT32_t', 'INT64_t', 'FLOAT32_t', 'FLOAT64_t', 'COMPLEX64_t', 'COMPLEX128_t']
INDEX_TYPES = ['INT32_t', 'INT64_t']

# when coding
#ELEMENT_TYPES = ['FLOAT64_t']
#INDEX_TYPES = ['INT32_t']

GENERAL_CONTEXT = {
                    'type_list': ELEMENT_TYPES,
                    'index_list' : INDEX_TYPES
                }

GENERAL_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader('/'), # we use absolute filenames
    trim_blocks=False,
    variable_start_string='@',
    variable_end_string='@')

GENERAL_ENVIRONMENT.filters['type2enum'] = type2enum


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

def generate_following_only_index(logger, template_filenames, template_environment, index_types, ext):
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

            generate_template_files(logger, template_filenames, template_environment, context, '_%s%s' % (index_name, ext))

def generate_following_type_and_index(logger, template_filenames, template_environment, element_types, index_types, ext, tabu_combination=None):
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
    for filename in template_filenames:

        for type_name in element_types:
            for index_name in index_types:
                if tabu_combination is not None:
                    try:
                        if not tabu_combination[type_name][index_name]:
                            continue
                    except:
                        pass

                context = {
                    'type'  : type_name,
                    'index' : index_name,
                    'type_list': ELEMENT_TYPES,
                    'index_list' : INDEX_TYPES
                }
                generate_template_files(logger, template_filenames, template_environment, context, '_%s_%s%s' % (index_name, type_name, ext))

#################################################################################################
# SETUP FILE
#################################################################################################
SETUP_FILE = os.path.join(PATH, 'setup.cpy')
SETUP_PY_FILE = os.path.join(PATH, 'setup.py')


#################################################################################################
# SPARSE
#################################################################################################
SPARSE_DIR = os.path.join(PATH, 'cysparse', 'sparse')

### include_utils
SPARSE_SPARSE_UTILS_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 'sparse_utils')
SPARSE_SPARSE_UTILS_DECLARATION_FILES = glob.glob(os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, '*.cpd'))
SPARSE_SPARSE_UTILS_DEFINITION_FILES = glob.glob(os.path.join(SPARSE_SPARSE_UTILS_TEMPLATE_DIR, '*.cpx'))

#################################################################################################
# SPARSE MATRICES
#################################################################################################

### SparseMatrix



SPARSE_MATRIX_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 'sparse_mat_matrices')

SPARSE_MATRIX_INCLUDE_FILES = glob.glob(os.path.join(SPARSE_MATRIX_TEMPLATE_DIR, '*.cpi'))
SPARSE_MATRIX_DECLARATION_FILES = glob.glob(os.path.join(SPARSE_MATRIX_TEMPLATE_DIR, '*.cpd'))
SPARSE_MATRIX_DEFINITION_FILES = glob.glob(os.path.join(SPARSE_MATRIX_TEMPLATE_DIR, '*.cpx'))


### LLSparseMatrix

LL_SPARSE_MATRIX_BASE_FILE = os.path.join(SPARSE_DIR, 'll_mat.cpx')

LL_SPARSE_MATRIX_TEMPLATE_DIR = os.path.join(SPARSE_DIR, 'll_mat_matrices')

LL_SPARSE_MATRIX_INCLUDE_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_TEMPLATE_DIR, '*.cpi'))
LL_SPARSE_MATRIX_DECLARATION_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_TEMPLATE_DIR, '*.cpd'))
LL_SPARSE_MATRIX_DEFINITION_FILES = glob.glob(os.path.join(LL_SPARSE_MATRIX_TEMPLATE_DIR, '*.cpx'))

### CSRSparseMatrix

### CSCSparseMatrix

### CSBSparseMatrix



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

    if arg_options.matrices or arg_options.all:
        action = True
        logger.info("Act for matrices")

        if arg_options.clean:
            logger.info("Clean generated files")
            clean_cython_files(logger, SPARSE_SPARSE_UTILS_TEMPLATE_DIR)
            clean_cython_files(logger, SPARSE_MATRIX_TEMPLATE_DIR)
            clean_cython_files(logger, LL_SPARSE_MATRIX_TEMPLATE_DIR)
            clean_cython_files(logger, SPARSE_DIR, [LL_SPARSE_MATRIX_BASE_FILE[:-4] + '.pyx'])

        else:
            logger.info("Generate code files")

            ###############################
            # Sparse
            ###############################
            ### include_utils
            generate_following_only_index(logger, SPARSE_SPARSE_UTILS_DECLARATION_FILES, GENERAL_ENVIRONMENT, INDEX_TYPES, '.pxd')
            generate_following_only_index(logger, SPARSE_SPARSE_UTILS_DEFINITION_FILES, GENERAL_ENVIRONMENT, INDEX_TYPES, '.pyx')

            ###############################
            # SparseMatrix
            ###############################
            # sparse_matrix_@index@_@type@.pxd and sparse_matrix_@index@_@type@.pyx
            generate_following_type_and_index(logger, SPARSE_MATRIX_DECLARATION_FILES, GENERAL_ENVIRONMENT, ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, SPARSE_MATRIX_DEFINITION_FILES, GENERAL_ENVIRONMENT, ELEMENT_TYPES, INDEX_TYPES, '.pyx')

            # TODO: add the possibility to use tabu in for loops in Jinja2 (i.e. inside a template file)....
            #tabu = {}
            #tabu['INT32_t'] = {}
            #tabu['INT32_t']['INT64_t'] = False

            ###############################
            # LLSparseMatrix
            ###############################
            # ll_mat.pyx
            generate_template_files(logger, [LL_SPARSE_MATRIX_BASE_FILE], GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.pyx')

            # ll_mat_@index@_@type@.pxd and ll_mat_@index@_@type@.pyx
            generate_following_type_and_index(logger, LL_SPARSE_MATRIX_DECLARATION_FILES, GENERAL_ENVIRONMENT, ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, LL_SPARSE_MATRIX_DEFINITION_FILES, GENERAL_ENVIRONMENT, ELEMENT_TYPES, INDEX_TYPES, '.pyx')

    if not action:
        logger.warning("No action proceeded...")

    logger.info("Stop some action(s)")
