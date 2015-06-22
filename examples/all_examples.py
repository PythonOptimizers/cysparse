import subprocess
import glob
import os

python_script_examples = glob.glob(os.path.join('*.py'))

for script in python_script_examples:
    if script in ['__init__.py', 'all_examples.py']:
        break
    print script
    subprocess.call("python %s" % script, shell=True)

