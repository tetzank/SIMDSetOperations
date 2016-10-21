#include <benchmark/benchmark.h>

#include "benchmark.hpp"

#include "union/naive.hpp"
#include "union/stl.hpp"
#include "union/branchless.hpp"
#include "union/sse.hpp"


BENCHMARK_TEMPLATE(BM_benchmark, union_scalar)->RangePair(8, 8<<10, 8, 8<<10);
BENCHMARK_TEMPLATE(BM_benchmark, union_scalar_stl)->RangePair(8, 8<<10, 8, 8<<10);
//BENCHMARK_TEMPLATE(BM_benchmark, intersect_scalar_stl_parallel)->RangePair(8, 8<<10, 8, 8<<10);
BENCHMARK_TEMPLATE(BM_benchmark, union_scalar_branchless)->RangePair(8, 8<<10, 8, 8<<10);

BENCHMARK_TEMPLATE(BM_benchmark, union_vector_sse)->RangePair(8, 8<<10, 8, 8<<10);


BENCHMARK_MAIN()
