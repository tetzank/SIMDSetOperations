#include <benchmark/benchmark.h>

#include "benchmark.hpp"


#include "difference/naive.hpp"
#include "difference/stl.hpp"
#include "difference/sse.hpp"


BENCHMARK_TEMPLATE(BM_benchmark, difference_scalar)->RangePair(8, 8<<10, 8, 8<<10);
BENCHMARK_TEMPLATE(BM_benchmark, difference_scalar_stl)->RangePair(8, 8<<10, 8, 8<<10);
//BENCHMARK_TEMPLATE(BM_benchmark, difference_scalar_stl_parallel)->RangePair(8, 8<<10, 8, 8<<10);

BENCHMARK_TEMPLATE(BM_benchmark, difference_vector_sse)->RangePair(8, 8<<10, 8, 8<<10);

BENCHMARK_MAIN()
