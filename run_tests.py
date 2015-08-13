import fnmatch
import os
import sys
import unittest
import subprocess

import shutil
import distutils

platform = distutils.util.get_platform()
python_version = sys.version

lib_dir =  "build" + os.path.sep + "lib." + platform + "-" + python_version[0:3]
destination_dir = lib_dir + os.path.sep + "tests"

# clean libxxx/tests because shutil.copytree only copies non existing directories
shutil.rmtree(destination_dir, ignore_errors=True)
shutil.copytree("tests", destination_dir, symlinks=False, ignore=None)

os.chdir(lib_dir)
subprocess.call(['nosetests','tests'])