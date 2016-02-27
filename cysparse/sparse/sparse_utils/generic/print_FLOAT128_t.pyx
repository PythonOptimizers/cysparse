#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
from cysparse.common_types.cysparse_types cimport *


cdef extern from "complex.h":
    float crealf(float complex z)
    float cimagf(float complex z)

    double creal(double complex z)
    double cimag(double complex z)

    long double creall(long double complex z)
    long double cimagl(long double complex z)

    double cabs(double complex z)
    float cabsf(float complex z)
    long double cabsl(long double complex z)

    double complex conj(double complex z)
    float complex  conjf (float complex z)
    long double complex conjl (long double complex z)

cdef extern from 'math.h':
    double fabs  (double x)
    float fabsf (float x)
    long double fabsl (long double x)

    double sqrt (double x)
    float sqrtf (float x)
    long double sqrtl (long double x)
    double log  (double x)


cdef element_to_string_FLOAT128_t(FLOAT128_t v, int cell_width=10):
    """
    Return a string representing an element of type FLOAT128_t.


    """
    # This is the *main* and *unique* function to print an element of a sparse matrix. All other printing functions
    # **must** use this function.
    cdef:
        FLOAT64_t exp



    exp = log(fabsl(v))




    if abs(exp) <= 4:
        if exp < 0:

            return ("%9.6f" % v).ljust(cell_width)


        else:

            return ("%9.*f" % (6,v)).ljust(cell_width)


    else:

        return ("%9.2e" % v).ljust(cell_width)



cdef conjugated_element_to_string_FLOAT128_t(FLOAT128_t v, int cell_width=10):
    """
    Return a string representing the conjugate of an element of type FLOAT128_t.

    Note:
        This function works for **all** types, not only complex ones.

    """

    # start to add the possibility of having to conjugate non complex elements...
    # TODO: see if we allow this or not. For the moment, this is only called from complex typed matrices.
    return element_to_string_FLOAT128_t(v)



cdef empty_to_string_FLOAT128_t(int cell_width=10):
    """
    return an empty cell in a matrix string representation

    """

    return "---".center(cell_width)
