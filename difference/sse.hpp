#ifndef DIFFERENCE_SSE_HPP_
#define DIFFERENCE_SSE_HPP_

#include <cstring>

#include <immintrin.h>

#include "naive.hpp"


size_t difference_vector_sse(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count = 0;
	size_t i_a = 0, i_b = 0;

	// trim lengths to be a multiple of 4
	size_t st_a = (size1 / 4) * 4;
	size_t st_b = (size2 / 4) * 4;

	__m128i cmp_mask = _mm_setzero_si128();
	while(i_a < st_a && i_b < st_b) {
		// load segments of four 32-bit elements
		__m128i v_a = _mm_load_si128((__m128i*)&list1[i_a]);
		__m128i v_b = _mm_load_si128((__m128i*)&list2[i_b]);

		// move pointers
// 		int32_t a_max = _mm_extract_epi32(v_a, 3);
		int32_t a_max = list1[i_a+3];
// 		int32_t b_max = _mm_extract_epi32(v_b, 3);
		int32_t b_max = list2[i_b+3];
		i_a += (a_max <= b_max) * 4;
		i_b += (a_max >= b_max) * 4;

		// not usable here but mentioned in paper: _mm_slli_si128
		// compute mask of common elements
		constexpr int32_t cyclic_shift = _MM_SHUFFLE(0,3,2,1); //rotating right
		constexpr int32_t cyclic_shift2= _MM_SHUFFLE(2,1,0,3); //rotating left
		constexpr int32_t cyclic_shift3= _MM_SHUFFLE(1,0,3,2); //between
		__m128i cmp_mask1 = _mm_cmpeq_epi32(v_a, v_b);         // pairwise comparison
		__m128i rot1 = _mm_shuffle_epi32(v_b, cyclic_shift);   // shuffling
		__m128i cmp_mask2 = _mm_cmpeq_epi32(v_a, rot1);        // again...
		__m128i rot2 = _mm_shuffle_epi32(v_b, cyclic_shift3);
		__m128i cmp_mask3 = _mm_cmpeq_epi32(v_a, rot2);        // and again...
		__m128i rot3 = _mm_shuffle_epi32(v_b, cyclic_shift2);
		__m128i cmp_mask4 = _mm_cmpeq_epi32(v_a, rot3);        // and again.

		// above the same code as in intersection
		// for difference we have to save cmp_mask for following iterations
		// only write elements out when list1 is stepped forward
		cmp_mask = _mm_or_si128(cmp_mask,
			_mm_or_si128(
				_mm_or_si128(cmp_mask1, cmp_mask2),
				_mm_or_si128(cmp_mask3, cmp_mask4)
			) // OR-ing of comparison masks
		); // OR it with the previous iteration

		__m128i lt_mask = _mm_cmpgt_epi32(v_b, v_a); //operands switched, SSE has lt but AVX2 only has gt
		__m128i le_mask = _mm_or_si128(cmp_mask1, lt_mask);
		constexpr int32_t splat_last = _MM_SHUFFLE(3,3,3,3);
		le_mask = _mm_shuffle_epi32(le_mask, splat_last);
		// write mask
		__m128i wmask = _mm_andnot_si128(cmp_mask, le_mask);
		// reset cmp_mask if we have to
		cmp_mask = _mm_andnot_si128(le_mask, cmp_mask);

		// convert the 128-bit mask to the 4-bit mask
		int32_t mask = _mm_movemask_ps((__m128)wmask);

		// copy out common elements
		__m128i p = _mm_shuffle_epi8(v_a, shuffle_mask[mask]);
		_mm_storeu_si128((__m128i*)&result[count], p);
		count += _mm_popcnt_u32(mask); // a number of elements is a weight of the mask
	}

	// if cmp_mask isn't 0, then some elements were seen in SIMD code above
	// we have to skip them when processing the tail
	uint32_t mask = _mm_movemask_ps((__m128)cmp_mask);
	while(mask){
		if(mask & 1){
			// skip element in list1, was seen in SIMD code
			++i_a; mask >>= 1;
		}else{
			//FIXME: copy&paste of scalar code
			if(i_b == size2){
				while(i_a < size1){
					if((mask & 1) == 0){
						result[count++] = list1[i_a];
					}
					++i_a; mask >>=1;
				}
				return count + (size1 - i_a);
			}
			if(list1[i_a] < list2[i_b]){
				result[count++] = list1[i_a];
				++i_a; mask >>= 1;
			}else if(list1[i_a] > list2[i_b]){
				++i_b;
			}else{
				++i_a; mask >>= 1;
				++i_b;
			}
		}
	}
	// intersect the tail using scalar intersection
	count += difference_scalar(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

	return count;
}


#endif
