# Run this script every time you make a git flow release, AFTER the release
import subprocess
import shutil
import os


if __name__ == '__main__':
    # Copy locally the whole cysparse repository
    CYSPARSE_DIR = os.path.curdir
    CYSPARSE_DEST = os.path.abspath(os.path.join(CYSPARSE_DIR, '..', 'tmp_cysparse_34534'))

    shutil.rmtree(CYSPARSE_DEST, ignore_errors=True)
    shutil.copytree(CYSPARSE_DIR, CYSPARSE_DEST, symlinks=True, ignore=shutil.ignore_patterns('build', 'dist'))

    # go into new repository
    os.chdir(CYSPARSE_DEST)

    # Switch to develop branch
    subprocess.call(['git', 'checkout', 'develop'])

    # create local brand new branch
    BRANCH_NAME = 'minimal_source'
    subprocess.call(['git', 'checkout', '-b', BRANCH_NAME])

    # clean
    subprocess.call(['python', 'clean.py'])
    subprocess.call(['python', 'generate_code.py', '-ac'])

    # commit files
    subprocess.call(['git', 'add', '--all'])
    # retrieve last commit sha1
    p = subprocess.Popen(['git', 'rev-parse', 'develop'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    last_commit_str, err = p.communicate()
    return_code = p.returncode

    subprocess.call(['git', 'commit', '-m', 'minimal version of commit %s' % last_commit_str])
    # delete remote branch
    subprocess.call(['git', 'push', 'origin', '--delete', BRANCH_NAME])
    # create remote branch: -u = set-upstream and push
    subprocess.call(['git', 'push', '-u', 'origin', BRANCH_NAME])

    # clean temporary repository
    shutil.rmtree(CYSPARSE_DEST, ignore_errors=True)