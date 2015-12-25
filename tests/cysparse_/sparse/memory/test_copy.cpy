#!/usr/bin/env python

"""
This file tests the common ``copy()`` method for **all** sparse objects.

``is_symmetric`` returns if a matrix or matrix-like object is indeed symmetric or not.

Warning:
    ``destroy_symmetry()`` uses randomness. Randomness should better be avoided in test cases but we do an exception
    here as it should not have any impact on the correctness of the tests.

Note:
    Don't be confused with ``store_symmetry``.
"""


if __name__ == '__main__':
    unittest.main()


