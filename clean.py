import os
import shutil

# ad hoc script to clean repository
directories = ['build']
directories_to_skip = ['.git', '.idea', 'tests', 'benchmarks', 'doc']

files_exts = ['.c', '.so', '.pyc']


def scandirs(path):
    for root, dirs, files in os.walk(path):
        if any(root.endswith(name) for name in directories_to_skip):
            continue
        for currentFile in files:
            #print "processing file: " + currentFile
            if any(currentFile.lower().endswith(ext) for ext in files_exts):
                os.remove(os.path.join(root, currentFile))

########################################################################################################################
# main
########################################################################################################################

current_path = os.path.curdir

try:
    for dir in directories:
        shutil.rmtree(os.path.join(current_path, dir))
except:
    pass

try:
    scandirs(current_path)
except:
    pass

