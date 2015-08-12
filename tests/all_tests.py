import fnmatch
import os
import sys
import unittest
import subprocess

def all_test_modules(root_dir, pattern):
    test_file_names = all_files_in(root_dir, pattern)
    #return [path_to_module(str) for str in test_file_names]
    return test_file_names


def all_files_in(root_dir, pattern):
    """
    Fetches all files with a given pattern.

    Args:
        root_dir:
        pattern:

    """
    matches = []

    old_root = root_dir
    for root, dirnames, filenames in os.walk(root_dir):
        print root


        matches = []
        for filename in fnmatch.filter(filenames, pattern):
            print "\t", filename
            matches.append(filename)
        old_root = os.getcwd()
        os.chdir(root)
        for filename in matches:
            print "python %s" % filename
            subprocess.call(['python', filename])
        os.chdir(old_root)

    return matches

def path_to_module(py_file):
    return strip_leading_dots(replace_slash_by_dot(strip_extension(py_file)))

def strip_extension(py_file):
    return py_file[0:len(py_file) - len('.py')]

def replace_slash_by_dot(str):
    return str.replace('\\', '.').replace('/', '.')

def strip_leading_dots(str):
    while str.startswith('.'):
       str = str[1:len(str)]
    return str

module_names = all_test_modules('.', 'test*.py')

print module_names

sys.exit(0)

suites = [unittest.defaultTestLoader.loadTestsFromName(mname) for mname 
    in module_names]

testSuite = unittest.TestSuite(suites)
runner = unittest.TextTestRunner(verbosity=1)
runner.run(testSuite)