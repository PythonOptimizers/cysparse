import os
import sys
import subprocess

import shutil
import distutils

#TODO: add switches to regenerate code and cleaning

platform = distutils.util.get_platform()
python_version = sys.version

lib_dir =  "build" + os.path.sep + "lib." + platform + "-" + python_version[0:3]
destination_dir = lib_dir + os.path.sep + "tests"

# clean libxxx/tests because shutil.copytree only copies non existing directories
shutil.rmtree(destination_dir, ignore_errors=True)
shutil.copytree("tests", destination_dir, symlinks=False, ignore=None)

# generate lib
subprocess.call(['python', 'generate_code.py','-a'])
subprocess.call(['python', 'setup.py', 'build'])

# launch tests
os.chdir(lib_dir)
subprocess.call(['nosetests','tests'])