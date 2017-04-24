#ifndef BENCHMARK_HPP_
#define BENCHMARK_HPP_

#include <numeric>
#include <random>
#include <algorithm>

#include "shuffle_dictionary.hpp"


template<size_t (*func)(const uint32_t*,size_t,const uint32_t*,size_t,uint32_t*)>
static void BM_benchmark(benchmark::State &state){
	uint32_t *numberx = new uint32_t[state.range_x()]; //TODO: make larger, atm the same size as lists in the end
	uint32_t *numbery = new uint32_t[state.range_y()]; //      so shuffling doesn't do anything as it's sorted afterwards
	std::iota(numberx, numberx+state.range_x(), 0);
	std::iota(numbery, numbery+state.range_y(), 0);
	std::random_device rng;
	std::shuffle(numberx, numberx+state.range_x(), rng);
	std::shuffle(numbery, numbery+state.range_y(), rng);

	uint32_t *listx = (uint32_t*)aligned_alloc(64, state.range_x()*sizeof(uint32_t));
	if(!listx){ puts("bad alloc"); return; }
	uint32_t *listy = (uint32_t*)aligned_alloc(64, state.range_y()*sizeof(uint32_t));
	if(!listy){ puts("bad alloc"); return; }
	std::copy(numberx, numberx+state.range_x(), listx);
	std::sort(listx, listx+state.range_x());
	std::copy(numbery, numbery+state.range_y(), listy);
	std::sort(listy, listy+state.range_y());

	delete[] numberx;
	delete[] numbery;

	uint32_t *result = (uint32_t*)aligned_alloc(64, std::max(state.range_x(), state.range_y()) * sizeof(uint32_t));
	if(!result){ puts("bad alloc"); return; }
	while(state.KeepRunning()){
		benchmark::DoNotOptimize(
			func(listx, state.range_x(), listy, state.range_y(), result)
		);
	}
	free(result);

	free(listx);
	free(listy);
}

#endif
