#!/usr/bin/env python

"""
This script generates a template for unit tests for the :program:`CySparse` library.

The template can then be completed by hand.

"""
from generator import TestGenerator

import os
import sys
import argparse

TYPE_TEST_DICT = {0: 'matrices',
                  1: 'matrix-likes',
                  2: 'sparse-likes'}

TEST_FILE_EXTENSION = '.cpy'


#####################################################
# PARSER
#####################################################
def make_parser():
    """
    Create a comment line argument parser.

    Returns:
        The command line parser.
    """
    parser = argparse.ArgumentParser(description='%s: a unit test code generator for the CySparse library' % os.path.basename(sys.argv[0]))
    parser.add_argument("-t", "--test_type", help="Type of test (0: 'matrices', 1: 'matrix-likes', 2: 'sparse-likes')", type=int, required=True)
    parser.add_argument("-n", "--test_name", help="Name of test (used for class names and file name)", required=True)

    return parser


def generate_tests(test_name, test_type):
    test_type_name = TYPE_TEST_DICT[test_type]

    test_generator = TestGenerator(test_name=test_name, test_type=test_type_name)

    with open("test_" + test_name.lower() + TEST_FILE_EXTENSION, 'w') as f:
        test_generator.generate_test(OUTSTREAM=f)





if __name__ == '__main__':

    # line arguments
    parser = make_parser()
    arg_options = parser.parse_args()

    if not 0<=arg_options.test_type<=2:
        print "Test type not recognized (must be in [0..2])\n"
        parser.print_help()
        sys.exit(-1)

    generate_tests(arg_options.test_name, arg_options.test_type)
