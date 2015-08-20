import os
import shutil
import argparse
import subprocess

# ad hoc script to clean repository
directories = ['build']
directories_to_skip = ['.git', '.idea', 'tests', 'benchmarks', 'build', 'doc']

files_exts = ['.c', '.so', '.pyc']


def scandirs(path, untrack=False):
    for root, dirs, files in os.walk(path):
        for dir in directories_to_skip:
            if dir in dirs:
                dirs.remove(dir)
        for currentFile in files:
            if any(currentFile.lower().endswith(ext) for ext in files_exts):
                if untrack:
                    subprocess.call(["git", "rm", "--cached", "-q", os.path.join(root, currentFile).split(path+os.sep)[1]])
                else:
                    os.remove(os.path.join(root, currentFile))

########################################################################################################################
# main
########################################################################################################################

parser = argparse.ArgumentParser(description='Clean CySparse library from .so and .c files')
parser.add_argument("-u", "--untrack", help="Untrack files from git", action='store_true', required=False)
arg_options = parser.parse_args()
 
current_path = os.path.curdir

if not arg_options.untrack:
    try:
        for dir in directories:
            shutil.rmtree(os.path.join(current_path, dir))
    except:
        pass

try:
    if arg_options.untrack:
        files_exts.remove('.pyc')
        current_path = '.'
    scandirs(current_path, untrack=arg_options.untrack)
except:
    pass

