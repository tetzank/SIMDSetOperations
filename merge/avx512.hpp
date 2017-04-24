#ifndef MERGE_AVX512_HPP_
#define MERGE_AVX512_HPP_

#include <immintrin.h>

#include <algorithm>

#include "naive.hpp"

#include "../union/avx512.hpp" //for blend masks and shuffles


size_t merge_vector_avx512_bitonic2(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count = 0;
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
	size_t i_a = 0, i_b = 0;
	// trim lengths to be a multiple of 16
	size_t st_a = ((size1-1) / 16) * 16;
	size_t st_b = ((size2-1) / 16) * 16;
	//
	uint32_t a_nextfirst, b_nextfirst;
	alignas(64) uint32_t maxtail[16];

	if(i_a < st_a && i_b < st_b){
		// load all the shuffles
		__m512i vL1L2 = _mm512_load_epi32(shuffles2[0]);
		__m512i vL2L3 = _mm512_load_epi32(shuffles2[1]);
		__m512i vL3L4 = _mm512_load_epi32(shuffles2[2]);
		__m512i vL4L5 = _mm512_load_epi32(shuffles2[3]);
		__m512i vL5Out_L = _mm512_load_epi32(shuffles2[4]);
		__m512i vL5Out_H = _mm512_load_epi32(shuffles2[5]);

		__mmask16 kL1L2 = blendmasks[0];
		__mmask16 kL2L3 = blendmasks[1];
		__mmask16 kL3L4 = blendmasks[2];
		__mmask16 kL4L5 = blendmasks[3];

		__m512i vreverse = _mm512_load_epi32(reverseshuffle);

		__m512i v_a = _mm512_load_epi32(list1);
		__m512i vb = _mm512_load_epi32(list2);
		__m512i v_b = _mm512_permutexvar_epi32(vreverse, vb);

		do{
			// bitonic merge network
			// level 1
			__m512i min = _mm512_min_epi32(v_a, v_b);
			__m512i max = _mm512_max_epi32(v_a, v_b);
			__m512i L = _mm512_mask_blend_epi32(kL1L2, min, max);
			__m512i H = _mm512_permutex2var_epi32(min, vL1L2, max);
			// level 2
			min = _mm512_min_epi32(L, H);
			max = _mm512_max_epi32(L, H);
			L = _mm512_mask_blend_epi32(kL2L3, min, max);
			H = _mm512_permutex2var_epi32(min, vL2L3, max);
			// level 3
			min = _mm512_min_epi32(L, H);
			max = _mm512_max_epi32(L, H);
			L = _mm512_mask_blend_epi32(kL3L4, min, max);
			H = _mm512_permutex2var_epi32(min, vL3L4, max);
			// level 4
			min = _mm512_min_epi32(L, H);
			max = _mm512_max_epi32(L, H);
			L = _mm512_mask_blend_epi32(kL4L5, min, max);
			H = _mm512_permutex2var_epi32(min, vL4L5, max);
			// level 5
			min = _mm512_min_epi32(L, H);
			max = _mm512_max_epi32(L, H);
			L = _mm512_permutex2var_epi32(min, vL5Out_L, max);
			H = _mm512_permutex2var_epi32(min, vL5Out_H, max);

			_mm512_store_epi32(&result[count], L);
			count += 16;

			v_a = H;
			// compare first element of the next block in both lists
			a_nextfirst = list1[i_a+16];
			b_nextfirst = list2[i_b+16];
			// write minimum as above out to result
			// keep maximum and do the same steps as above with next block
			// next block from one list, which first element in new block is smaller
			i_a += (a_nextfirst <= b_nextfirst) * 16;
			i_b += (a_nextfirst >= b_nextfirst) * 16;
			size_t index = (a_nextfirst <= b_nextfirst)? i_a: i_b;
			const uint32_t *base = (a_nextfirst <= b_nextfirst)? list1: list2;
			v_b = _mm512_load_epi32(&base[index]);
		}while(i_a < st_a && i_b < st_b);
		
		// v_a contains max vector from last comparison, v_b contains new, might be out of bounds
		// indices i_a and i_b correct, still need to handle v_a
		_mm512_store_epi32(maxtail, _mm512_permutexvar_epi32(vreverse, v_a));

		size_t mti=0;
		size_t mtsize = 16;
		if(a_nextfirst <= b_nextfirst){
			// compare maxtail with list1
			while(mti < mtsize && i_a < size1){
				if(maxtail[mti] < list1[i_a]){
					result[count++] = maxtail[mti];
					mti++;
				}else{
					result[count++] = list1[i_a];
					i_a++;
				}
			}
			i_b += 16;
		}else{
			// compare maxtail with list2
			while(mti < mtsize && i_b < size2){
				if(maxtail[mti] < list2[i_b]){
					result[count++] = maxtail[mti];
					mti++;
				}else{
					result[count++] = list2[i_b];
					i_b++;
				}
			}
			i_a += 16;
		}
		while(mti < mtsize){
			result[count++] = maxtail[mti++];
		}
	}

	// scalar tail
	count += merge_scalar(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

#endif
	return count;
}

#endif
