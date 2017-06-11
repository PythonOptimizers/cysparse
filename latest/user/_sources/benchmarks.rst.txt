.. Benchmarks

Benchmarks
==========

In the following benchmarks, we compare CySparse with
pysparse `\cite{pysparse-web}` and and we only discuss
multiplication of a sparse matrix with a dense vector. We randomly
generate square sparse matrices of size :math:`n \times n` with
:math:`nnz` nonzero elements. We generate dense vectors with ’s
``arange(0, n, dtype=np.float64)``.

For each benchmark, we run the four scenarii described in
Table [tab:4\_benchmark\_scenarii] :math:`100` times for each operation.
For each scenario, operations are ranked by runtime. The most efficient
implementation gets a value of 1.0. The values of the other operations
are relative, i.e. an operation with value :math:`k` takes :math:`k`
times as long to execute as the most efficient operation.

+----------+----------+-----------+
| Scenario | `n`      |  `nnz`    |
+----------+----------+-----------+
| 1        | `10^4`   | `10^3`    |
+----------+----------+-----------+
| 2        | `10^5`   | `10^4`    |
+----------+----------+-----------+
| 3        | `10^6`   | `10^5`    |
+----------+----------+-----------+
| 4        | `10^6`   | `5 10^3`  |
+----------+----------+-----------+

\cysparse, \pysparse and \scipy were compiled and tested with the same flags on the same machine
and using ``int32`` indices and ``float64`` elements.

Table [tab:sparse\_matrix\_dense\_numpy\_vector\_multiplication] reports
benchmarks on the basic ``matvec`` operation, i.e. the multiplication
:math:`A v` where :math:`A` is a :math:`n \times n` sparse matrix and
:math:`v` is a vector of length :math:`n`.

lets the user simply type ``A*v`` to compute the product. does not
support the notation ``A*v`` and only offers ``A.matvec(v,y)`` where
``y`` is a preallocated vector to store the result. Because both and
allocate such a vector transparently, we take into account the time
required by to allocate ``y``. [1]_ and  2 in
Table [tab:sparse\_matrix\_dense\_numpy\_vector\_multiplication]
represent ``y = A*v`` and ``y = A.matvec(v)`` respectively while and  2
represent ``y = A*v`` and ``y = A.‘_mul‘_vector(v)``, respectively. The
“2” variants are equivalent to ``A*v`` minus a small convenience layer
that permits the shorthand notation.

+------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                    y = A*v with \texttt{A} a LL sparse matrix                                                  |
+------------------------------------+-----------------------------------+------------------------------------+----------------------------------+
|            Scenario 1              |             Scenario 2            |             Scenario 3             |            Scenario 4            |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| pysparse        |           1.000  | pysparse        |          1.000  | cysparse 2      |           1.000  | pysparse        |          1.000 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| cysparse 2      |           1.158  | cysparse 2      |          1.023  | cysparse        |           1.023  | cysparse        |          1.043 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| cysparse        |           1.220  | cysparse        |          1.032  | pysparse        |           1.045  | cysparse 2      |          1.064 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| scipy 2         |          86.395  | scipy           |        101.536  | scipy 2         |          55.690  | scipy           |        133.495 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| scipy           |          87.267  | scipy 2         |        102.131  | scipy           |          56.398  | scipy 2         |        135.512 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+

+------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                    y = A*v with \texttt{A} a CSR sparse matrix                                                 |
+------------------------------------+-----------------------------------+------------------------------------+----------------------------------+
|            Scenario 1              |             Scenario 2            |             Scenario 3             |            Scenario 4            |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| pysparse        |           1.000  | cysparse 2      |          1.000  | cysparse        |           1.000  | pysparse        |          1.000 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| cysparse 2      |           1.178  | cysparse        |          1.022  | pysparse        |           1.000  | cysparse        |          1.018 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| cysparse        |           1.212  | pysparse        |          1.037  | cysparse 2      |           1.009  | cysparse 2      |          1.061 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| scipy 2         |           1.333  | scipy 2         |          1.287  | scipy 2         |           1.114  | scipy           |          1.419 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| scipy           |           1.373  | scipy           |          1.308  | scipy           |           1.135  | scipy 2         |          1.421 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+

+------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                    y = A*v with \texttt{A} a CSC sparse matrix                                                 |
+------------------------------------+-----------------------------------+------------------------------------+----------------------------------+
|            Scenario 1              |             Scenario 2            |             Scenario 3             |            Scenario 4            |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| scipy 2         |           1.000  | scipy           |          1.000  | scipy 2         |           1.000  | scipy           |          1.000 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| scipy           |           1.102  | scipy 2         |          1.004  | scipy           |           1.002  | scipy 2         |          1.000 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| cysparse 2      |           1.107  | cysparse        |          1.084  | cysparse        |           1.059  | cysparse        |          1.138 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| cysparse        |           1.146  | cysparse 2      |          1.120  | cysparse 2      |           1.060  | cysparse 2      |          1.148 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+


Table [tab:sparse\_matrix\_dense\_numpy\_vector\_multiplication] reveals
that using LL and CSR formats, and are on par while is slightly faster
than when using the CSC format.

The second benchmark in
Table [tab:csc\_sparse\_matrix\_dense\_numpy\_vector\_multiplication]
investigates the case where the dense vector is not contiguous in
memory. We can see that our specialized implementation pays off.

+------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                $w = A\cdot v$ with $A$ a CSC sparse matrix and $v$ is non contiguous                           |
+------------------------------------+-----------------------------------+------------------------------------+----------------------------------+
|            Scenario 1              |             Scenario 2            |             Scenario 3             |            Scenario 4            |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| cysparse 2      |           1.000  | cysparse 2      |          1.000  | cysparse 2      |           1.000  | cysparse        |          1.000 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| cysparse        |           1.011  | cysparse        |          1.098  | cysparse        |           1.006  | cysparse 2      |          1.007 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| scipy 2         |           1.354  | scipy           |          2.273  | scipy 2         |           1.733  | scipy 2         |          3.193 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| scipy           |           1.394  | scipy 2         |          2.286  | scipy           |           1.742  | scipy           |          3.216 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+

Table: CSC sparse matrix with dense non contiguous vector multiplication

The third and last benchmark in
Table [tab:sparse\_matrix2\_dense\_numpy\_vector\_multiplication]
compares the multiplication of two sparse matrices with a dense vector.
computes :math:`A \cdot B \cdot v` as :math:`A \cdot (B \cdot v)` and
clearly this is faster than first computing :math:`A \cdot B` and then
:math:`(A \cdot B) \cdot v`. Even when we force to compute
:math:`A \cdot (B \cdot v)`, remains slightly faster.

+------------------------------------------------------------------------------------------------------------------------------------------------+
|                   y = A*B*v with \texttt{A} a CSR sparse matrix, $B$ a CSC sparse matrix and \texttt{v} a dense \numpy vector                  |
+------------------------------------+-----------------------------------+------------------------------------+----------------------------------+
|            Scenario 1              |             Scenario 2            |             Scenario 3             |            Scenario 4            |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| cysparse        |           1.000  | cysparse        |          1.000  | cysparse        |           1.000  | cysparse        |          1.000 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| scipy           |           5.386  | scipy           |          3.921  | scipy           |           3.850  | scipy           |          5.207 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+


+------------------------------------------------------------------------------------------------------------------------------------------------+
|                  y = A*(B*v) with \texttt{A} a CSR sparse matrix, $B$ a CSC sparse matrix and \texttt{v} a dense \numpy vector                 |
+------------------------------------+-----------------------------------+------------------------------------+----------------------------------+
|            Scenario 1              |             Scenario 2            |             Scenario 3             |            Scenario 4            |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| cysparse        |           1.000  | cysparse        |          1.000  | cysparse        |           1.000  | cysparse        |          1.000 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+
| scipy           |           1.019  | scipy           |          1.016  | scipy           |           1.011  | scipy           |          1.049 |
+-----------------+------------------+-----------------+-----------------+-----------------+------------------+-----------------+----------------+


.. [1]
   Using ``y = numpy.empty(n, dtype=numpy.float64)``.
