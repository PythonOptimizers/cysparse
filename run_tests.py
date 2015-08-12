#import unittest

# testmodules = [
#     'cysparse.linalg.suitesparse.umfpack.generic.test_umfpack',
#     'cysparse.test_whiteutils',
#     'cysparse.test_cogapp',
#     ]
#
# suite = unittest.TestSuite()
#
# for t in testmodules:
#     try:
#         # If the module defines a suite() function, call it to get the suite.
#         mod = __import__(t, globals(), locals(), ['suite'])
#         suitefn = getattr(mod, 'suite')
#         suite.addTest(suitefn())
#     except (ImportError, AttributeError):
#         # else, just load all the test cases from the module.
#         suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))
#
# unittest.TextTestRunner().run(suite)

#!/usr/bin/python

# import unittest2 as unittest
#
# if __name__ == "__main__":
#     all_tests = unittest.TestLoader().discover('tests', pattern='*.py')
#     unittest.TextTestRunner().run(all_tests)


import unittest
import tests.all_tests
testSuite = tests.all_tests.create_test_suite()
text_runner = unittest.TextTestRunner().run(testSuite)