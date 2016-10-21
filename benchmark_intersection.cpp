#include <benchmark/benchmark.h>

#include "benchmark.hpp"


#include "intersection/naive.hpp"
#include "intersection/stl.hpp"
#include "intersection/branchless.hpp"
#include "intersection/sse.hpp"
#include "intersection/avx.hpp"
#include "intersection/avx2.hpp"


BENCHMARK_TEMPLATE(BM_benchmark, intersect_scalar)->RangePair(8, 8<<10, 8, 8<<10);
BENCHMARK_TEMPLATE(BM_benchmark, intersect_scalar_stl)->RangePair(8, 8<<10, 8, 8<<10);
//BENCHMARK_TEMPLATE(BM_benchmark, intersect_scalar_stl_parallel)->RangePair(8, 8<<10, 8, 8<<10);
BENCHMARK_TEMPLATE(BM_benchmark, intersect_scalar_branchless)->RangePair(8, 8<<10, 8, 8<<10);

BENCHMARK_TEMPLATE(BM_benchmark, intersect_vector_sse)->RangePair(8, 8<<10, 8, 8<<10);
BENCHMARK_TEMPLATE(BM_benchmark, intersect_vector_sse_asm)->RangePair(8, 8<<10, 8, 8<<10);

BENCHMARK_TEMPLATE(BM_benchmark, intersect_vector_avx)->RangePair(8, 8<<10, 8, 8<<10);
BENCHMARK_TEMPLATE(BM_benchmark, intersect_vector_avx2)->RangePair(8, 8<<10, 8, 8<<10);


BENCHMARK_MAIN()
