from cysparse.types.cysparse_types cimport *


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


cdef element_to_string_COMPLEX256_t(COMPLEX256_t v, int cell_width=10):
    """
    Return a string representing an element of type COMPLEX256_t.


    """
    # This is the *main* and *unique* function to print an element of a sparse matrix. All other printing functions
    # **must** use this function.
    cdef:
        FLOAT64_t exp

        FLOAT128_t real_part, imag_part
    sign = '+'



    exp = log(cabsl(v))



    
    real_part = creall(v)
    imag_part = cimagl(v)
    
    if imag_part < 0.0:
        sign = '-'
        imag_part *= -1.0


    if abs(exp) <= 4:
        if exp < 0:

            return ("%9.6f" % real_part).ljust(cell_width) + sign + ("%9.6fj" % imag_part).ljust(cell_width)


        else:

            return ("%9.*f" % (6, real_part)).ljust(cell_width) + sign + ("%9.*fj" % (6, imag_part)).ljust(cell_width)


    else:

        return ("%9.2e" % real_part).ljust(cell_width) + sign + ("%9.2ej" % imag_part).ljust(cell_width)



cdef conjugated_element_to_string_COMPLEX256_t(COMPLEX256_t v, int cell_width=10):
    """
    Return a string representing the conjugate of an element of type COMPLEX256_t.

    Note:
        This function works for **all** types, not only complex ones.

    """

    return element_to_string_COMPLEX256_t(conjl(v), cell_width)



cdef empty_to_string_COMPLEX256_t(int cell_width=10):
    """
    return an empty cell in a matrix string representation

    """

    return "---".center(cell_width) + ' ' + "---".center(cell_width)
