"""
Factory method to access SPQR.

The python code (this module) is autmotically generated because the code depends
on the compile/architecture configuration.

"""

from cysparse.sparse.ll_mat import PyLLSparseMatrix_Check
from cysparse.cysparse_types.cysparse_types import *

{% for index_type in spqr_index_list %}
    {% for element_type in spqr_type_list %}
from cysparse.linalg.suitesparse.spqr.spqr_@index_type@_@element_type@ import SPQRContext_@index_type@_@element_type@
    {% endfor %}
{% endfor %}

def NewSPQRContext(A, verbose=False):
    """
    Create and return the right SPQR context object.

    Args:
        A: :class:`LLSparseMatrix`.
    """
    if not PyLLSparseMatrix_Check(A):
        raise TypeError('Matrix A should be a LLSparseMatrix')

    itype = A.itype
    dtype = A.dtype

{% for index_type in spqr_index_list %}
    {% if index_type == spqr_index_list |first %}
    if itype == @index_type|type2enum@:
    {% for element_type in spqr_type_list %}
        {% if element_type == spqr_type_list |first %}
        if dtype == @element_type|type2enum@:
        {% else %}
        elif dtype == @element_type|type2enum@:
        {% endif %}
            return SPQRContext_@index_type@_@element_type@(A, verbose=verbose)
    {% endfor %}
    {% else %}
    elif itype == @index_type|type2enum@:
    {% for element_type in spqr_type_list %}
        {% if element_type == spqr_type_list |first %}
        if dtype == @element_type|type2enum@:
        {% else %}
        elif dtype == @element_type|type2enum@:
        {% endif %}
            return SPQRContext_@index_type@_@element_type@(A, verbose=verbose)
    {% endfor %}
    {% endif %}
{% endfor %}

    allowed_types = '\titype:
    {%- for index_name in spqr_index_list -%}
       @index_name|type2enum@
       {%- if index_name != spqr_index_list|last -%}
       ,
       {%- endif -%}
     {%- endfor -%}
     \n\tdtype:
     {%- for element_name in spqr_type_list -%}
       @element_name|type2enum@
       {%- if element_name != spqr_type_list|last -%}
       ,
       {%- endif -%}
     {%- endfor -%}
     \n'

    type_error_msg = 'Matrix has an index and/or element type that is incompatible with SPQR\nAllowed types:\n%s' % allowed_types
    raise TypeError(type_error_msg)
