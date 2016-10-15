#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cassert>

#include <immintrin.h>


#if __GNUC__ >= 5
#include <parallel/algorithm>
#endif

#include "constants.h"


size_t union_scalar(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t counter=0;
	const uint32_t *end1 = list1+size1, *end2 = list2+size2;
	while(list1 != end1 && list2 != end2){
		if(*list1 < *list2){
			result[counter++] = *list1;
			list1++;
		}else if(*list1 > *list2){
			result[counter++] = *list2;
			list2++;
		}else{
			result[counter++] = *list1;
			list1++; list2++;
		}
	}
	// copy rest, can't be the same
	memcpy(result+counter, list1, (end1-list1)*sizeof(uint32_t));
	counter += end1 - list1;
	memcpy(result+counter, list2, (end2-list2)*sizeof(uint32_t));
	counter += end2 - list2;

	return counter;
}

size_t union_scalar_stl(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	uint32_t *endresult = std::set_union(list1, list1+size1, list2, list2+size2, result);
	return endresult-result;
}

size_t union_scalar_stl_parallel(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
#if __GNUC__ >= 5
	uint32_t *endresult = std::__parallel::set_union((uint32_t*)list1, (uint32_t*)(list1+size1), (uint32_t*)list2, (uint32_t*)(list2+size2), result);
	return endresult-result;
#else
	return 0;
#endif
}

size_t union_scalar_branchless(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	const uint32_t *end1 = list1+size1, *end2 = list2+size2;
	uint32_t *endresult=result;
#ifndef NOASM
	asm(".intel_syntax noprefix;"
		"xor rax, rax;"
		"xor rbx, rbx;"
	"1: "
		"cmp %[list1], %[end1];"  // list1 != end1
		"je 2f;"
		"cmp %[list2], %[end2];"  // list2 != end2
		"je 2f;"

		"mov r11d, [%q[list1]];"  // move both to registers as cmov only works on registers
		"mov r10d, [%q[list2]];"  // saved in r10d as value is only 4 byte wide

		"cmp r11d, r10d;"   // compare *list1 and *list2
		"setbe al;"         // set al=1 if lower or equal
		"setae bl;"         // set bl=1 if greater or equal

		"cmovb r10d, r11d;" // save *list1 instead of *list2, if lower
		"mov [%q[endresult]], r10d;"  // always save

		"lea %q[list1], [%q[list1] + rax*4];"     // list1++, if lower or equal
		"lea %q[list2], [%q[list2] + rbx*4];"     // list2++, if greater or equal
		"lea %q[endresult], [%q[endresult] + 4];" // result++, always

		"jmp 1b;"       // to loop head
// 		"cmp %1, %3;"  // list1 != end1
// 		"je 2f;"
// 		"cmp %2, %4;"  // list2 != end2
// 		"jne 1b;"
	"2: "
		".att_syntax;"

		: [endresult]"=r"(endresult), [list1]"=r"(list1), [list2]"=r"(list2)
		: [end1]"r"(end1), [end2]"r"(end2),
			"0"(endresult), "1"(list1), "2"(list2)
		: "%rax","%rbx", "%r10","%r11", "memory", "cc"
	);
#endif
	// copy rest, can't be the same
	memcpy(endresult, list1, (end1-list1)*sizeof(uint32_t));
	endresult += end1 - list1;
	memcpy(endresult, list2, (end2-list2)*sizeof(uint32_t));
	endresult += end2 - list2;

	return endresult-result;
}

// a simple implementation, we don't care about performance here
static __m128i shuffle_mask[16]; // precomputed dictionary
void prepare_shuffling_dictionary() {
	for(int i = 0; i < 16; i++) {
		int counter = 0;
		char permutation[16];
		memset(permutation, 0xFF, sizeof(permutation));
		for(char b = 0; b < 4; b++) {
			if(i & (1 << b)) { // get bit b from i
				permutation[counter++] = 4*b;
				permutation[counter++] = 4*b + 1;
				permutation[counter++] = 4*b + 2;
				permutation[counter++] = 4*b + 3;
			}
		}
		__m128i mask = _mm_loadu_si128((const __m128i*)permutation);
		shuffle_mask[i] = mask;
	}
}

size_t union_sse(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count = 0;
#ifdef __SSE2__
	size_t i_a = 0, i_b = 0;
	// trim lengths to be a multiple of 4
	size_t st_a = ((size1-1) / 4) * 4;
	size_t st_b = ((size2-1) / 4) * 4;
	uint32_t endofblock=~0, a_nextfirst, b_nextfirst;
	uint32_t maxtail[4];

	if(i_a < st_a && i_b < st_b){
		__m128i v_a = _mm_load_si128((__m128i*)&list1[i_a]);
		__m128i v_b = _mm_load_si128((__m128i*)&list2[i_b]);

		do {
			__m128i step1min = _mm_min_epu32(v_a, v_b);
			__m128i step1max = _mm_max_epu32(v_a, v_b);

			constexpr int32_t cyclic_shift = _MM_SHUFFLE(2,1,0,3);
			__m128i tmp = _mm_shuffle_epi32(step1max, cyclic_shift);
			__m128i step2min = _mm_min_epu32(step1min, tmp);
			__m128i step2max = _mm_max_epu32(step1min, tmp);

			__m128i tmp2 = _mm_shuffle_epi32(step2max, cyclic_shift);
			__m128i step3min = _mm_min_epu32(step2min, tmp2);
			__m128i step3max = _mm_max_epu32(step2min, tmp2);

			__m128i tmp3 = _mm_shuffle_epi32(step3max, cyclic_shift);
			__m128i step4min = _mm_min_epu32(step3min, tmp3);
			__m128i step4max = _mm_max_epu32(step3min, tmp3);

			__m128i tmp4 = _mm_shuffle_epi32(step4max, cyclic_shift);

			// deduplicate over block end, 1 2 3 4 | 4 5 6 7
			uint32_t first = _mm_extract_epi32(step4min, 0);
			count -= (endofblock==first);
			endofblock = _mm_extract_epi32(step4min, 3);
			// in register deduplicate, only removes inside one vector
			__m128i dedup = _mm_shuffle_epi32(step4min, cyclic_shift);
			__m128i dedup_mask = _mm_cmpeq_epi32(step4min, dedup);
			// flip mask
			dedup_mask = _mm_andnot_si128(dedup_mask, _mm_cmpeq_epi32(tmp, tmp));
			// compress shuffle like in intersect
			// convert the 128-bit mask to the 4-bit mask
			int32_t mask = _mm_movemask_ps((__m128)dedup_mask);
			__m128i p = _mm_shuffle_epi8(step4min, shuffle_mask[mask]);
			_mm_storeu_si128((__m128i*)&result[count], p);
			count += _mm_popcnt_u32(mask); // a number of elements is a weight of the mask

			v_a = tmp4;
			// compare first element of the next block in both lists
			a_nextfirst = list1[i_a+4];
			b_nextfirst = list2[i_b+4];
			// write minimum as above out to result
			// keep maximum and do the same steps as above with next block
			// next block from one list, which first element in new block is smaller
			asm(".intel_syntax noprefix;"

				"xor rax, rax;"
				"xor rbx, rbx;"
				"cmp %5, %6;"
				"setbe al;"
				"seta  bl;"
				"lea %q0, [%q0 + rax*4];"
				"lea %q1, [%q1 + rbx*4];"

				// load next block from list with smaller first element
				"mov r10, %q8;"
				"mov r11, %q1;"
				"cmovbe r10, %q7;"
				"cmovbe r11, %q0;"
				"vmovdqa %2, [r10 + r11*4];" // this might read past the end of one array, not used afterwards as loop head fails

				".att_syntax"
				: "=r"(i_a), "=r"(i_b), "=x"(v_b)
				: "0"(i_a), "1"(i_b), "r"(a_nextfirst), "r"(b_nextfirst), "r"(list1), "r"(list2)
				: "%eax","%ebx", "%r10","%r11", "cc"
			);
		}while(i_a < st_a && i_b < st_b);
		// v_a contains max vector from last comparison, v_b contains new, might be out of bounds
		// indices i_a and i_b correct, still need to handle v_a
		_mm_storeu_si128((__m128i*)maxtail, v_a);

		size_t mti=0;
		size_t mtsize = std::unique(maxtail, maxtail+4) - maxtail; // deduplicate tail
		if(a_nextfirst <= b_nextfirst){
			// endofblock needs to be considered too, for deduplication
			if(endofblock == std::min(maxtail[0],list1[i_a])) --count;
			// compare maxtail with list1
	// 		count += union_scalar_branchless(maxtail, 4, list1+i_a, size1-i_a, result+count);
			while(mti < mtsize && i_a < size1){
				if(maxtail[mti] < list1[i_a]){
					result[count++] = maxtail[mti];
					mti++;
				}else if(maxtail[mti] > list1[i_a]){
					result[count++] = list1[i_a];
					i_a++;
				}else{
					result[count++] = maxtail[mti];
					mti++; i_a++;
				}
			}
			i_b += 4;
		}else{
			// endofblock needs to be considered too, for deduplication
			if(endofblock == std::min(maxtail[0],list2[i_b])) --count;
			// compare maxtail with list2
	// 		count += union_scalar_branchless(maxtail, 4, list2+i_b, size2-i_b, result+count);
			while(mti < mtsize && i_b < size2){
				if(maxtail[mti] < list2[i_b]){
					result[count++] = maxtail[mti];
					mti++;
				}else if(maxtail[mti] > list2[i_b]){
					result[count++] = list2[i_b];
					i_b++;
				}else{
					result[count++] = maxtail[mti];
					mti++; i_b++;
				}
			}
			i_a += 4;
		}
		while(mti < mtsize){
			result[count++] = maxtail[mti++];
		}
	}

	// scalar tail
	count += union_scalar_branchless(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);
#endif

	return count;
}


void run(uint32_t **lists, size_t (*func)(const uint32_t*,size_t,const uint32_t*,size_t,uint32_t*)){
	uint32_t *union_list = (uint32_t*)aligned_alloc(32, 2*arraySize*sizeof(uint32_t));
	auto t_start = std::chrono::high_resolution_clock::now();
	size_t union_count=0;
	for(size_t i=0; i<listCount; ++i){
		for(size_t j=i+1; j<listCount; ++j){
			union_count += func(lists[i], arraySize, lists[j], arraySize, union_list);
		}
	}
	auto t_end = std::chrono::high_resolution_clock::now();
	printf("Wall clock time passed: %10.2f ms - %lu\n",
		std::chrono::duration<double, std::milli>(t_end-t_start).count(),
		union_count
	);
	free(union_list);
}

int main(){
#if 1
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
	std::cout << "preparing lists done - "
		<< std::chrono::duration<double, std::milli>(t_end-t_start).count()<< " ms"
		<< std::endl;


	puts("naive scalar union:");
	run(lists, union_scalar);
	puts("stl set_union:");
	run(lists, union_scalar_stl);
#if __GNUC__ >= 5
	puts("stl parallel set_union: uses more than one core, just for reference here");
	run(lists, union_scalar_stl_parallel);
#endif

	puts("branchless scalar union:");
	run(lists, union_scalar_branchless);

#ifdef __SSE2__
	prepare_shuffling_dictionary();
	puts("SSE union:");
	run(lists, union_sse);
#endif


	// cleanup
	for(size_t i=0; i<listCount; ++i){
		free(lists[i]);
	}
	delete[] lists;
#else

#if 0
	const uint32_t list1[] = {
		 1,  3,  5,  7,  9, 11, 13, 15, 17, 19,
		21, 23, 25, 27, 29, 31, 33, 35, 37, 39
	};
	constexpr size_t list1Size = sizeof(list1)/sizeof(list1[0]);
	const uint32_t list2[] = {
		 2,  4,  6,  8, 10, 12, 14, 16, 18, 20,
		22, 24, 26, 28, 30, 32, 34, 36, 38, 40
	};
	constexpr size_t list2Size = sizeof(list2)/sizeof(list2[0]);
	uint32_t result[list1Size + list2Size];
#endif

	const unsigned int size = 23;
	uint32_t *list1 = (uint32_t*)aligned_alloc(32, size*sizeof(uint32_t));
	uint32_t *list2 = (uint32_t*)aligned_alloc(32, size*sizeof(uint32_t));
// 	for(unsigned int i=0; i<size; ++i){
// 		list1[i] = i*2;    // even
// 		list2[i] = i*2 /*+1*/; // odd
// 	}
	uint32_t *result = (uint32_t*)aligned_alloc(32, 2*size*sizeof(uint32_t));
	const size_t list1Size=size, list2Size=size;

// 	for(size_t r=0; r<100000; ++r){
		for(unsigned int i=0; i<size; ++i){
			list1[i] = i*2;    // even
			list2[i] = i*2 /*+1*/; // odd
		}

// 		size_t count = union_scalar_branchless(list1, list1Size, list2, list2Size, result);
		prepare_shuffling_dictionary();
// 		size_t count = union_sse(list1, list1Size, list2, list2Size, result);
// 		size_t count = union_sse_asm(list1, list1Size, list2, list2Size, result);
		size_t count = union_sse_asm2(list1, list1Size, list2, list2Size, result);
		for(size_t i=0; i<count; ++i){
			printf("%u\n", result[i]);
		}
// 	}
#endif


	return 0;
}
