#ifndef UNION_SSE_HPP_
#define UNION_SSE_HPP_

#include <immintrin.h>

#include <algorithm>

#include "branchless.hpp"


size_t union_vector_sse(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
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
				"vmovdqa %2, [r10 + r11*4];" //FIXME: this might read past the end of one array, not used afterwards as loop head fails

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

#endif
