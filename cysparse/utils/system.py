import os
import inspect


def get_root_dir():
    thisfile = inspect.getabsfile(inspect.currentframe())
    thispath = os.path.dirname(thisfile)
    toppath  = os.path.dirname(thispath)
    return toppath
