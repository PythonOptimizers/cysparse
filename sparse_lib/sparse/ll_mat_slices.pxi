# TODO: delete this file


from cpython cimport PyObject

from lib

# TODO: use more internal CPython code
cdef extern from "Python.h":
    int PySlice_Check(PyObject *ob)


cdef long* create_indexlist(long *len, long maxlen, PyObject *A):
    cdef:
        long *index;
        long  i, j;

        Py_ssize_t start, stop, step, length;
        PyObject *val;

    # Integer
    if PyInt_Check(A):
        i = PyInt_AS_LONG(A)
        index = calloc(1, sizeof(long))
        if index:
            index[0] = i
        &len[0] = 1

        return index


    # Slice
    if PySlice_Check(A):

        if PySlice_GetIndicesEx(<PySliceObject*>A, maxlen, &start, &stop, &step, &length) < 0:
            return NULL

    index = calloc(length, sizeof(long))
    if index:
        for j from 0 <= j < length:
            i=start
            index[j] = i

            i += step

        &len[0] = <long>length
        return index





    return index