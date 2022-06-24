from benchopt.benchmark import Benchmark
from benchopt import run_benchmark

bench_aa = Benchmark('./benchmark_aa_single')

run_benchmark(bench_aa, max_runs=25, n_jobs=1, n_repetitions=1)
