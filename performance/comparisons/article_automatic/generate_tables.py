"""
Little script to generate the three tables 2, 3 and 4 for the article

Cahier du GERAD G-2016-00
nlp.py: An Object-Oriented Environment for
Large-Scale Optimization


"""

# LaTeX tables

TABLE_BEGIN = r"""\begin{table}
\begin{center}
\begin{tabular}"""

TABLE_END = r"""\end{tabular}
\end{center}
 \caption{%s}\label{%s}
\end{table}"""

TABLE_DOUBLE_ARGUMENT = 'lr'
TABLE_SEPARATOR = '|'

HLINE = r'\hline'


# Bench table
BENCHMARK_REPORT_TITLE = 'Benchmark Report'

BENCHMARK_END = 'Each of the above'


#\multicolumn{8}{c}{$w = A\cdot v$ with $A$ a LL sparse matrix}\\
#\hline
#\multicolumn{2}{c}{Scenario 1} & \multicolumn{2}{c}{Scenario 2} & \multicolumn{2}{c}{Scenario 3} & \multicolumn{2}{c}{Scenario 4}\\
#\hline
#\pysparse   &            $1.0$    &  \cysparse    &           $1.0$  & \cysparse 2  &           $1.0$  &  \cysparse 2   &              $1.0$ \\
#\cysparse 2 &  $1.16849173554$    &  \pysparse    & $1.01144839655$  & \cysparse    & $1.00671434468$  &  \pysparse     &    $1.04245103382$ \\
#\cysparse   &  $1.18780991736$    &  \cysparse 2  & $1.0412044374$   & \pysparse    & $1.00875086382$  &  \cysparse     &    $1.17517514888$ \\
#\scipy 2    &  $95.5508264463$    &  \scipy 2     & $72.7128294244$  & \scipy 2     & $56.5681294727$  &  \scipy 2      &    $147.931627916$ \\
#\scipy      &  $97.1334710744$    &  \scipy       & $73.4207209247$  & \scipy       & $57.4677389582$  &  \scipy        &    $151.740297047$ \\

def generate_numbers(scenarii, filename, nbr_of_test):
    """
    Generate the numbers from the benchmark reports.

    Args:
        scenarii: List with pair of scenarii.
        filename: bench filename.
        nbr_of_test: For each scenario, the number of different routines tested.
        translate_dic: dict to translate from benchmark test name to its LaTeX equivalent.

    """
    file_is_benchmark = False
    inside_scenario = False
    inside_title = False

    nbr_of_scenarii = len(scenarii)
    nbr_of_scenarii_read = 0

    nbr_of_fields_in_a_scenario = 6

    nbr_of_test_read = 0

    # tables
    cols_count = nbr_of_scenarii
    rows_count = nbr_of_test
    tests_table = [['' for x in range(cols_count)] for x in range(rows_count)]
    number_table = [['' for x in range(cols_count)] for x in range(rows_count)]

    with open(filename) as f:
        for index, line in enumerate(f):

            stripped_line = line.strip()

            empty_line = stripped_line == ''

            if not file_is_benchmark:
                if stripped_line.startswith(BENCHMARK_REPORT_TITLE):
                    file_is_benchmark = True
                    inside_title = True
                    continue

            if inside_title:
                #print "inside title"
                # each title is followed by an underline
                inside_title = False
                continue

            if inside_scenario:

                #print "inside scenario"
                #print stripped_line

                data = stripped_line.split('|')
                assert len(data) == nbr_of_fields_in_a_scenario

                tests_table[nbr_of_test_read][nbr_of_scenarii_read] = data[0].strip()
                number_table[nbr_of_test_read][nbr_of_scenarii_read] = data[5].strip()

                nbr_of_test_read += 1

                if nbr_of_test_read == nbr_of_test:
                    nbr_of_test_read = 0
                    inside_scenario = False
                    nbr_of_scenarii_read += 1
                continue

            # find out if we are in a title or in a scenario
            if not empty_line:
                scenario_title = stripped_line.split('|')
                if len(scenario_title) == nbr_of_fields_in_a_scenario:
                    inside_title = True
                    inside_scenario = True
                    continue

            if stripped_line.startswith(BENCHMARK_END):
                #print "end of file"
                break

            if empty_line:
                #print "empty line"
                # line is empty: reset markers
                inside_scenario = False
                inside_title = False

        return file_is_benchmark and nbr_of_scenarii_read == nbr_of_scenarii, tests_table, number_table

def generate_sub_table(scenarii, caption, filename, nbr_of_test, translate_dic, field_width=16):
    """
    Generate the numbers from the benchmark reports.

    Args:
        scenarii: List with pair of scenarii.
        caption: Sub caption of the benchmark.
        filename: bench filename.
        nbr_of_test: For each scenario, the number of different routines tested.
        translate_dic: dict to translate from benchmark test name to its LaTeX equivalent.
        field_width: Width of each field.

    """
    bench_mark_file, table_names, table_numbers = generate_numbers(scenarii, filename, nbr_of_test)

    assert bench_mark_file, "%s is not a benchmark file or does not correspond to the given scenarii..." % filename

    nbr_of_scenarii = len(scenarii)

    cols_count = nbr_of_scenarii
    rows_count = nbr_of_test



    table = []
    table.append(r'\multicolumn{%d}{c}{%s}\\' % (2 * nbr_of_scenarii, caption))
    table.append(HLINE)

    scenario_title_table = []
    for i in range(nbr_of_scenarii):
        scenario_title_table.append(r'\multicolumn{2}{c}{Scenario %d}' % (i + 1))

    table.append(' & '.join(scenario_title_table) + r'\\')
    table.append(HLINE)

    # data (numbers)
    for test_name in range(nbr_of_test):
        # construct line by line
        line_of_data_table = []
        for scenario in range(nbr_of_scenarii):
            #first the test name
            line_of_data_table.append('{name: <{fill}}'.format(name=translate_dic[table_names[test_name][scenario]], fill=field_width))
            #second the number
            number = '$%s$' % table_numbers[test_name][scenario]
            line_of_data_table.append('{name: >{fill}}'.format(name=number, fill=field_width))

        table.append(' & '.join(line_of_data_table) + r'\\')
    return '\n'.join(table)


def generate_table(scenarii, caption, label, sub_tables, translate_dic, field_width=16):

    table = []

    nbr_of_scenarii = len(scenarii)


    table.append(TABLE_BEGIN + '{' + TABLE_SEPARATOR.join([TABLE_DOUBLE_ARGUMENT] * nbr_of_scenarii) + '}')

    for sub_table in sub_tables:
        filename, subtitle, nbr_of_tests = sub_table
        table.append(HLINE)
        table.append(generate_sub_table(scenarii=scenarii,
                             caption=subtitle,
                             filename=filename,
                             nbr_of_test=nbr_of_tests,
                             translate_dic=translate_dic,
                             field_width=field_width))
        table.append(HLINE)

    table.append(TABLE_END % (caption, label))

    return '\n'.join(table)

if __name__ == "__main__":

    # see article
    benchmark_scenarii = [(10000,1000), (100000,10000), (1000000,100000), (1000000,5000)]
    routines_name_translation_dict = {
        'pysparse' : r'\pysparse',
        'cysparse': r'\cysparse',
        'cysparse2': r'\cysparse 2',
        'scipy sparse': r'\scipy',
        'scipy sparse2': r'\scipy 2'
    }
    nbr_of_scenarii = len(benchmark_scenarii)

    suffix = '.bench.txt'
    field_width = 16

    # Table 2:
    print "=" * 80
    print "Table 2:"
    print
    first_subtitle = r'$w = A\cdot v$ with $A$ a LL sparse matrix'
    first_filename = 'matvec_ll' + suffix

    second_subtitle = r'$w = A\cdot v$ with $A$ a CSR sparse matrix'
    second_filename = 'matvec_csr' + suffix

    third_subtitle = r'$w = A\cdot v$ with $A$ a CSC sparse matrix'
    third_filename = 'matvec_csc' + suffix

    sub_tables = [(first_filename, first_subtitle, 5), (second_filename, second_subtitle, 5), (third_filename, third_subtitle, 4)]

    print generate_table(scenarii=benchmark_scenarii,
                         caption=r'Benchmarks on a sparse matrix - dense \numpy vector multiplication',
                         label=r'tab:sparse_matrix_dense_numpy_vector_multiplication',
                         sub_tables=sub_tables,
                         translate_dic=routines_name_translation_dict, field_width=field_width)

    # Table 3:
    print "=" * 80
    print "Table 3:"
    print
    first_subtitle = r'$w = A\cdot v$ with $A$ a CSC sparse matrix and $v$ is non contiguous'
    first_filename = 'matvec_csc_non_contiguous' + suffix

    sub_tables = [(first_filename, first_subtitle, 4)]

    print generate_table(scenarii=benchmark_scenarii,
                         caption=r'Benchmarks on a CSC sparse matrix - dense non contiguous \numpy vector multiplication',
                         label=r'tab:csc_sparse_matrix_dense_numpy_vector_multiplication',
                         sub_tables=sub_tables,
                         translate_dic=routines_name_translation_dict, field_width=field_width)


    # Table 4:
    print "=" * 80
    print "Table 4:"
    print
    first_subtitle = r'$w = A \cdot B \cdot v$ with $A$ a CSR sparse matrix, $B$ a CSC sparse matrix and $v$ a dense \numpy vector'
    first_filename = '2mat_mul_vector' + suffix

    second_subtitle = r'$w = A \cdot (B \cdot v)$ with $A$ a CSR sparse matrix, $B$ a CSC sparse matrix and $v$ a dense \numpy vector'
    second_filename = '2mat_mul_vector_order_changed' + suffix

    sub_tables = [(first_filename, first_subtitle, 2), (second_filename, second_subtitle, 2)]

    print generate_table(scenarii=benchmark_scenarii,
                         caption=r'Benchmarks on a sparse matrix - dense \numpy vector multiplication where the sparse matrix is obtained by the multiplication of two sparse matrices',
                         label=r'tab:sparse_matrix2_dense_numpy_vector_multiplication',
                         sub_tables=sub_tables,
                         translate_dic=routines_name_translation_dict, field_width=field_width)
