#include <cstdio>
#include <chrono>

#include "projectconfig.h"
#include "constants.h"
#include "shuffle_dictionary.hpp"

#include "intersection/naive.hpp"
#include "intersection/stl.hpp"
#include "intersection/branchless.hpp"
#include "intersection/sse.hpp"
#include "intersection/avx.hpp"
#include "intersection/avx2.hpp"
#include "intersection/avx512.hpp"
#include "intersection/galloping.hpp"
#include "intersection/galloping_sse.hpp"
#include "intersection/galloping_avx2.hpp"



void run(uint32_t **lists,
	size_t (*func)(const uint32_t*,size_t,const uint32_t*,size_t,uint32_t*)=nullptr,
	size_t (*func_count)(const uint32_t*,size_t,const uint32_t*,size_t)=nullptr,
	size_t (*func_index)(const uint32_t*,size_t,const uint32_t*,size_t,uint32_t*)=nullptr
){
	if(func){
		auto t_start = std::chrono::high_resolution_clock::now();
		size_t intersected=0;
		for(size_t repeat=0; repeat<repeatCount; ++repeat){
			for(size_t i=0; i<listCount; ++i){
				uint32_t *intersected_list = new uint32_t[arraySize];
				for(size_t j=i+1; j<listCount; ++j){
					intersected += func(
						lists[i], arraySize,
						lists[j], arraySize,
						intersected_list
					);
				}
				delete[] intersected_list;
			}
		}
		auto t_end = std::chrono::high_resolution_clock::now();
		printf("Wall clock time passed: %10.2f ms - %lu\n",
			std::chrono::duration<double, std::milli>(t_end-t_start).count(),
			intersected
		);
	}

	if(func_count){
		auto t_start = std::chrono::high_resolution_clock::now();
		size_t intersected=0;
		for(size_t repeat=0; repeat<repeatCount; ++repeat){
			for(size_t i=0; i<listCount; ++i){
				for(size_t j=i+1; j<listCount; ++j){
					intersected += func_count(
						lists[i], arraySize,
						lists[j], arraySize
					);
				}
			}
		}
		auto t_end = std::chrono::high_resolution_clock::now();
		printf("Wall clock time passed: %10.2f ms - %lu - just counting\n",
			std::chrono::duration<double, std::milli>(t_end-t_start).count(),
			intersected
		);
	}
	if(func_index){
		auto t_start = std::chrono::high_resolution_clock::now();
		size_t intersected=0;
		for(size_t repeat=0; repeat<repeatCount; ++repeat){
			for(size_t i=0; i<listCount; ++i){
				uint32_t *intersected_list = new uint32_t[arraySize];
				for(size_t j=i+1; j<listCount; ++j){
					intersected += func(
						lists[i], arraySize,
						lists[j], arraySize,
						intersected_list
					);
				}
				delete[] intersected_list;
			}
		}
		auto t_end = std::chrono::high_resolution_clock::now();
		printf("Wall clock time passed: %10.2f ms - %lu - index\n",
			std::chrono::duration<double, std::milli>(t_end-t_start).count(),
			intersected
		);
	}
}

int main(){
	auto t_start = std::chrono::high_resolution_clock::now();
	uint32_t **lists = new uint32_t*[listCount];
	// load lists from file which was generated by genLists
	FILE *fd = fopen("test.dat", "rb");
	if(!fd){
		puts("couldn't open test.dat");
		return -1;
	}
	for(size_t i=0; i<listCount; ++i){
		lists[i] = (uint32_t*)aligned_alloc(32, arraySize*sizeof(uint32_t));
		fread(lists[i], 4, arraySize, fd);
	}
	fclose(fd);
	auto t_end = std::chrono::high_resolution_clock::now(); 
	printf("preparing lists done - %f ms\n",
		std::chrono::duration<double, std::milli>(t_end-t_start).count()
	);

	//puts("scalar:");
	//run(lists, intersect_scalar, intersect_scalar_count, intersect_scalar_index);
	puts("stl set_intersection:");
	run(lists, intersect_scalar_stl);
#if __GNUC__ >= 5
	puts("stl parallel set_intersection: uses more than one core, just for reference here");
	run(lists, intersect_scalar_stl_parallel);
#endif
	puts("c branchless scalar:");
	run(lists, intersect_scalar_branchless_c, intersect_scalar_branchless_c_count);

	puts("asm branchless scalar:");
	run(lists, intersect_scalar_branchless, intersect_scalar_branchless_count);

#ifdef __SSE2__
	prepare_shuffling_dictionary();
	puts("128bit SSE vector:");
	run(lists, intersect_vector_sse, intersect_vector_sse_count);
	puts("128bit SSE vector - asm:");
	run(lists, intersect_vector_sse_asm);
#endif

#ifdef __AVX__
	prepare_shuffling_dictionary_avx();

	puts("256bit AVX vector: (not AVX2)");
	run(lists, intersect_vector_avx, intersect_vector_avx_count);
	puts("256bit AVX vector: (not AVX2) - asm");
	//FIXME: normal intersection segfaults
	//run(lists, intersect_vector_avx_asm, intersect_vector_avx_asm_count);
	run(lists, nullptr, intersect_vector_avx_asm_count); 

#ifdef __AVX2__
	puts("256bit AVX2 vector");
	run(lists, intersect_vector_avx2, intersect_vector_avx2_count);
	puts("256bit AVX2 vector - asm");
	run(lists, intersect_vector_avx2_asm, intersect_vector_avx2_asm_count);
#endif

	free(shuffle_mask_avx);
#endif

#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
	puts("512bit AVX512 vector");
	run(lists, intersect_vector_avx512_conflict);
	puts("512bit AVX512 vector - asm");
	run(lists, intersect_vector_avx512_conflict_asm);
#endif

#if 0
	puts("v1");
	run(lists, v1);
	puts("v3");
	run(lists, v3);
	puts("SIMD galloping");
	run(lists, SIMDgalloping);
	puts("v1_avx2");
	run(lists, v1_avx2);
	puts("v3_avx2");
	run(lists, v3_avx2);
	puts("SIMD galloping AVX2");
	run(lists, SIMDgalloping_avx2);
#endif
#if 0
	puts("galloping SSE");
	run(lists, SIMDintersection);
	puts("galloping AVX2");
	run(lists, SIMDintersection_avx2);
#endif

//	puts("SIMD Galloping V1: AVX");
//	run(lists, intersect_galloping_V1_AVX, intersect_galloping_V1_AVX_count);
// 	puts("SIMD Galloping V1: SSE4.2"); //FIXME: broken
// 	run(lists, intersect_galloping_V1_SSE, intersect_galloping_V1_SSE_count);

	for(size_t i=0; i<listCount; ++i){
		free(lists[i]);
	}
	delete[] lists;

	return 0;
}
