cdef bint update_ll_mat_item_add(LLSparseMatrix A, INT_t i, INT_t j, FLOAT_t x):
    """
    Update-add matrix entry: ``A[i,j] += x``

    Args:
        A: Matrix to update.
        i, j: Coordinates of item to update.
        x (FLOAT_t): Value to add to item to update ``A[i, j]``.

    Returns:
        True.

    Raises:
        ``IndexError`` when non writing to lower triangle of a symmetric matrix.
    """
    cdef:
        INT_t k, new_elem, col, last

    if A.is_symmetric and i < j:
        raise IndexError("Write operation to upper triangle of symmetric matrix not allowed")

    if not A.store_zeros and x == 0.0:
        return True

    # Find element to be updated
    col = last = -1
    k = A.root[i]
    while k != -1:
        col = A.col[k]
        if col >= j:
            break
        last = k
        k = A.link[k]

    if col == j:
        # element already exists: compute updated value
        x += A.val[k]

        if A.store_zeros and x == 0.0:
            #  the updated element is zero and must be removed

            # relink row i
            if last == -1:
                A.root[i] = A.link[k]
            else:
                A.link[last] = A.link[k]

            # add element to free list
            A.link[k] = A.free
            A.free = k

            A.nnz -= 1
        else:
            A.val[k] = x
    else:
        # new item
        if A.free != -1:
            # use element from the free chain
            new_elem = A.free
            A.free = A.link[new_elem]
        else:
            # append new element to the end
            new_elem = A.nnz

            # test if there is space for a new element
            if A.nnz == A.nalloc:
                A._realloc_expand()

        A.val[new_elem] = x
        A.col[new_elem] = j
        A.link[new_elem] = k
        if last == -1:
            A.root[i] = new_elem
        else:
            A.link[last] = new_elem
        A.nnz += 1

    return True
