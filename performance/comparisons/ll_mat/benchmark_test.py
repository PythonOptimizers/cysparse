import benchmark


########################################################################################################################
# Benchmark
########################################################################################################################
class BenchmarkBenchmark(benchmark.Benchmark):


    label = "Test of benchmark"
    each = 30

    def eachSetUp(self):

        self.number_1 = 5
        self.number_2 = 5


    def tearDown(self):
        assert self.number_1 == self.number_2

    def test_pysparse(self):
        self.number_1 = 0
        return

    def test_cysparse(self):
        self.number_2 = 0
        return

if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")