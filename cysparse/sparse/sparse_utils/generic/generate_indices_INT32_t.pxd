from cysparse.types.cysparse_types cimport *
from cpython cimport PyObject

cdef INT32_t * create_c_array_indices_from_python_object_INT32_t(INT32_t max_length, PyObject * obj, INT32_t * number_of_elements) except NULL