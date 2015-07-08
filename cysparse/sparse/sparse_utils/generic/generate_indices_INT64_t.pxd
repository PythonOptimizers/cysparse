from cysparse.types.cysparse_types cimport *
from cpython cimport PyObject

cdef INT64_t * create_c_array_indices_from_python_object_INT64_t(INT64_t max_length, PyObject * obj, INT64_t * number_of_elements) except NULL