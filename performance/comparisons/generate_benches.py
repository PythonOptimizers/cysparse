import subprocess
import glob
import os
import sys



def generate_txt_benches(sub_dir, suffix='bench.txt'):
    python_script_benches = glob.glob(os.path.join(sub_dir, '*.py'))

    for script in python_script_benches:
        if script in ['__init__.py', 'generate_benches.py']:
            break
        print script
        subprocess.call("python %s > %s" % (script, script[:-3] + '.' + suffix), shell=True)

if __name__ == "__main__":

    generate_txt_benches(sys.argv[1])