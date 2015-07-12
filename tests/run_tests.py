
import unittest2
loader = unittest2.TestLoader()
tests = loader.discover('.')
testRunner = unittest2.runner.TextTestRunner()
testRunner.run(tests)