"""
Several helpers to get a package version.

Versioning: from https://packaging.python.org/en/latest/single_source_version.html#single-sourcing-the-version
(see also https://github.com/pypa/pip)
"""

import io
import os
import re


def read(absolute_base_path, *names, **kwargs):
    """
    Read a file in with utf8 encoding.

    Returns:
        File handle.

    Args:
        names:
        kwargs:

    Warning:
        Use **absolute** path for ``absolute_base_path``.

    """
    with io.open(
        os.path.join(os.path.dirname(absolute_base_path), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    """
    Find a version ``__version__`` in a given file.

    Args:
        file_paths: List of paths to join.

    Warning:
        Uses ``regex``.
    """
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")
