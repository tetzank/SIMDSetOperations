#ifndef DIFFERENCE_AVX2_HPP_
#define DIFFERENCE_AVX2_HPP_

#include <immintrin.h>

#if IACA
#include <iacaMarks.h>
#endif


size_t difference_vector_avx2(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
#ifdef __AVX2__
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;

	__m256i cmp_mask = _mm256_setzero_si256();
	__m256i splat_last = _mm256_set1_epi32(7);
#if IACA_DIFFERENCE_AVX2
	IACA_START
#endif
	while(i_a < st_a && i_b < st_b){
		__m256i v_a = _mm256_load_si256((__m256i*)&list1[i_a]);
		__m256i v_b = _mm256_load_si256((__m256i*)&list2[i_b]);

// 		int32_t a_max = _mm256_extract_epi32(v_a, 7);
		int32_t a_max = list1[i_a+7];
// 		int32_t b_max = _mm256_extract_epi32(v_b, 7);
		int32_t b_max = list2[i_b+7];
		i_a += (a_max <= b_max) * 8;
		i_b += (a_max >= b_max) * 8;

		constexpr int32_t cyclic_shift = _MM_SHUFFLE(0,3,2,1); //rotating right
		constexpr int32_t cyclic_shift2= _MM_SHUFFLE(2,1,0,3); //rotating left
		constexpr int32_t cyclic_shift3= _MM_SHUFFLE(1,0,3,2); //between
		__m256i cmp_mask1 = _mm256_cmpeq_epi32(v_a, v_b);
		__m256 rot1 = _mm256_permute_ps((__m256)v_b, cyclic_shift);
		__m256i cmp_mask2 = _mm256_cmpeq_epi32(v_a, (__m256i)rot1);
		__m256 rot2 = _mm256_permute_ps((__m256)v_b, cyclic_shift3);
		__m256i cmp_mask3 = _mm256_cmpeq_epi32(v_a, (__m256i)rot2);
		__m256 rot3 = _mm256_permute_ps((__m256)v_b, cyclic_shift2);
		__m256i cmp_mask4 = _mm256_cmpeq_epi32(v_a, (__m256i)rot3);

		__m256 rot4 = _mm256_permute2f128_ps((__m256)v_b, (__m256)v_b, 1);

		__m256i cmp_mask5 = _mm256_cmpeq_epi32(v_a, (__m256i)rot4);
		__m256 rot5 = _mm256_permute_ps(rot4, cyclic_shift);
		__m256i cmp_mask6 = _mm256_cmpeq_epi32(v_a, (__m256i)rot5);
		__m256 rot6 = _mm256_permute_ps(rot4, cyclic_shift3);
		__m256i cmp_mask7 = _mm256_cmpeq_epi32(v_a, (__m256i)rot6);
		__m256 rot7 = _mm256_permute_ps(rot4, cyclic_shift2);
		__m256i cmp_mask8 = _mm256_cmpeq_epi32(v_a, (__m256i)rot7);

		// above the same code as in intersection
		// for difference we have to save cmp_mask for following iterations
		// only write elements out when list1 is stepped forward
		cmp_mask = _mm256_or_si256(cmp_mask,
			_mm256_or_si256(
				_mm256_or_si256(
					_mm256_or_si256(cmp_mask1, cmp_mask2),
					_mm256_or_si256(cmp_mask3, cmp_mask4)
				),
				_mm256_or_si256(
					_mm256_or_si256(cmp_mask5, cmp_mask6),
					_mm256_or_si256(cmp_mask7, cmp_mask8)
				)
			)
		);// OR it with the previous iteration
	
		__m256i lt_mask = _mm256_cmpgt_epi32(v_b, v_a); //operands switched, SSE has lt but AVX2 only has gt
		__m256i le_mask = _mm256_or_si256(cmp_mask1, lt_mask);
		le_mask = _mm256_permutevar8x32_epi32(le_mask, splat_last);
		// write mask
		__m256i wmask = _mm256_andnot_si256(cmp_mask, le_mask);
		// reset cmp_mask if we have to
		cmp_mask = _mm256_andnot_si256(le_mask, cmp_mask);

		// convert the 256-bit mask to the 8-bit mask
		int32_t mask = _mm256_movemask_ps((__m256)wmask);

		__m256i idx = _mm256_load_si256((const __m256i*)&shuffle_mask_avx[mask*8]);
		__m256i p = _mm256_permutevar8x32_epi32(v_a, idx);
		_mm256_storeu_si256((__m256i*)&result[count], p);

		count += _mm_popcnt_u32(mask);
	}
#if IACA_DIFFERENCE_AVX2
	IACA_END
#endif

	// if cmp_mask isn't 0, then some elements were seen in SIMD code above
	// we have to skip them when processing the tail
	uint32_t mask = _mm256_movemask_ps((__m256)cmp_mask);
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

#endif
	return count;
}

#endif
