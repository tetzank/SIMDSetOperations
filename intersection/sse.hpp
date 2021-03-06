#ifndef INTERSECTION_SSE_HPP_
#define INTERSECTION_SSE_HPP_

#include <cstring>

#include <immintrin.h>

#include "naive.hpp"


// taken from: https://highlyscalable.wordpress.com/2012/06/05/fast-intersection-sorted-lists-sse/
size_t intersect_vector_sse(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count = 0;
	size_t i_a = 0, i_b = 0;

	// trim lengths to be a multiple of 4
	size_t st_a = (size1 / 4) * 4;
	size_t st_b = (size2 / 4) * 4;

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
		__m128i cmp_mask = _mm_or_si128(
				_mm_or_si128(cmp_mask1, cmp_mask2),
				_mm_or_si128(cmp_mask3, cmp_mask4)
		); // OR-ing of comparison masks
		// convert the 128-bit mask to the 4-bit mask
		int32_t mask = _mm_movemask_ps((__m128)cmp_mask);

		// copy out common elements
		__m128i p = _mm_shuffle_epi8(v_a, shuffle_mask[mask]);
		_mm_storeu_si128((__m128i*)&result[count], p);
		count += _mm_popcnt_u32(mask); // a number of elements is a weight of the mask
	}

	// intersect the tail using scalar intersection
	count += intersect_scalar(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

	return count;
}

size_t intersect_vector_sse_count(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2){
	size_t count = 0;
	size_t i_a = 0, i_b = 0;

	// trim lengths to be a multiple of 4
	size_t st_a = (size1 / 4) * 4;
	size_t st_b = (size2 / 4) * 4;

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
		__m128i cmp_mask = _mm_or_si128(
				_mm_or_si128(cmp_mask1, cmp_mask2),
				_mm_or_si128(cmp_mask3, cmp_mask4)
		); // OR-ing of comparison masks
		// convert the 128-bit mask to the 4-bit mask
		int32_t mask = _mm_movemask_ps((__m128)cmp_mask);

		count += _mm_popcnt_u32(mask); // a number of elements is a weight of the mask
	}

	// intersect the tail using scalar intersection
	count += intersect_scalar_count(list1+i_a, size1-i_a, list2+i_b, size2-i_b);

	return count;
}

#ifndef DISABLE_ASM
size_t intersect_vector_sse_asm(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count = 0;
	size_t i_a = 0, i_b = 0;

	// trim lengths to be a multiple of 4
	size_t st_a = (size1 / 4) * 4;
	size_t st_b = (size2 / 4) * 4;

	while(i_a < st_a && i_b < st_b) {
		asm(".intel_syntax noprefix;"

			"movdqa xmm0, [%q3 + %q[i_a]*4];" //__m128i v_a = _mm_load_si128((__m128i*)&list1[i_a]);
			"movdqa xmm1, [%q4 + %q[i_b]*4];" //__m128i v_b = _mm_load_si128((__m128i*)&list2[i_b]);

			"mov r8d, [%q[list1] + %q[i_a]*4 + 12];" //int32_t a_max = list1[i_a+3];
			"mov r9d, [%q[list2] + %q[i_b]*4 + 12];" //int32_t b_max = list2[i_b+3];
			//i_a += (a_max <= b_max) * 4;
			//i_b += (a_max >= b_max) * 4;
			"xor rax, rax;"
			"xor rbx, rbx;"
			"cmp r8d, r9d;" // shared comparison of a_max and b_max
			"setbe al;"     // a_max <= b_max
			"setae bl;"     // a_max >= b_max
			"lea %q[i_a], [%q[i_a] + rax*4];"
			"lea %q[i_b], [%q[i_b] + rbx*4];"

			"pshufd xmm2, xmm1, 0x39;" // rotating right
			"pshufd xmm3, xmm1, 0x4e;" // rotating left
			"pshufd xmm4, xmm1, 0x93;" // between
			// do all comparisons
			"pcmpeqd xmm1, xmm0;"
			"pcmpeqd xmm2, xmm0;"
			"pcmpeqd xmm3, xmm0;"
			"pcmpeqd xmm4, xmm0;"
			// OR-ing of comparison masks
			"por xmm2, xmm1;"
			"por xmm4, xmm3;"
			"por xmm2, xmm4;"

			"movmskps r8d, xmm2;" // movemask
			// save in result
			// 16 multiplier not possible in index expression
			// -> do it outside
			"movslq r9, r8d;"
			"shl r9, 4;"
			"pshufb xmm0, [%q[shuffle_mask] + r9];"    // compress
			"movups [%q[result] + %q[count]*4], xmm0;" // store

			// count += popcount(mask)
			"popcnt r8d, r8d;"
			"add %q[count], r8;"

			".att_syntax;"
			: [i_a]"=r"(i_a), [i_b]"=r"(i_b), [count]"=r"(count)
			: [list1]"r"(list1), [list2]"r"(list2), [result]"r"(result), [shuffle_mask]"r"(shuffle_mask),
				"0"(i_a), "1"(i_b), "2"(count)
			: "%rax", "%rbx", "%r8", "%r9",
				"%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4",
				"memory", "cc"
		);
	}
	// intersect the tail using scalar intersection
	count += intersect_scalar(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

	return count;
}
#endif

#endif
