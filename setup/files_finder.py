# Several helpers to find files and/or directories


def find_files(directory, pattern, recursively=True, complete_filename=True):
    """
    Return a list of files with or without base directories, recursively or not.

    Args:
        directory: base directory to start the search.
        pattern: fnmatch pattern for filenames.
        complete_filename: return complete filename or not?
        recursively: do we recurse or not?
    """

    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                if complete_filename:
                    filename = os.path.join(root, basename)
                else:
                    filename = basename
                yield filename
        if not recursively:
            break
