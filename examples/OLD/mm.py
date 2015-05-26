from cysparse.sparse.ll_mat import MakeLLSparseMatrix

import sys

if __name__ == "__main__":
    filename = sys.argv[1]

    C = MakeLLSparseMatrix(mm_filename=filename)

    print C
    C.print_to(sys.stdout)

    print "Virtual memory: " + str(C.memory_virtual())
    print "Real memory: " + str(C.memory_real())