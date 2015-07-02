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
    cdef:
        FLOAT64_t exp


    exp = log(cabsl(v))

    if abs(exp) <= 4:
        if exp < 0:

            return ("%9.6f" % creall(v)).ljust(cell_width) + '+' + ("%9.6fj" % cimagl(v)).ljust(cell_width)


        else:

            return ("%9.*f" % (6,creall(v))).ljust(cell_width) + '+' + ("%9.*fj" % (6,cimagl(v))).ljust(cell_width)


    else:

        return ("%9.2e" % creall(v)).ljust(cell_width) + '+' + ("%9.2ej" % cimagl(v)).ljust(cell_width)



cdef empty_to_string_COMPLEX256_t(int cell_width=10):
    """
    return an empty cell in a matrix string representation

    """

    return "---".center(cell_width) + ' ' + "---".center(cell_width)
