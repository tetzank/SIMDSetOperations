#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cassert>

//FIXME: i don't know when this was added, check for my current compiler version
#if __GNUC__ == 5 && __GNUC_MINOR__ == 2
#include <parallel/algorithm>
#endif

#include "projectconfig.h"

#include "constants.h"


size_t intersect_scalar(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
	size_t counter=0;
	uint32_t *end1 = list1+size1, *end2 = list2+size2;
	while(list1 != end1 && list2 != end2){
		if(*list1 < *list2){
			list1++;
		}else if(*list1 > *list2){
			list2++;
		}else{
			result[counter++] = *list1;
			list1++; list2++;
		}
	}
	return counter;
}
size_t intersect_scalar_count(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2){
	size_t counter=0;
	uint32_t *end1 = list1+size1, *end2 = list2+size2;
	while(list1 != end1 && list2 != end2){
		if(*list1 < *list2){
			list1++;
		}else if(*list1 > *list2){
			list2++;
		}else{
			counter++;
			list1++; list2++;
		}
	}
	return counter;
}

size_t intersect_scalar_stl(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
	uint32_t *endresult = std::set_intersection(list1, list1+size1, list2, list2+size2, result);
	return endresult-result;
}
size_t intersect_scalar_stl_parallel(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
//FIXME: i don't know when this was added, check for my current compiler version
#if __GNUC__ == 5 && __GNUC_MINOR__ == 2
	uint32_t *endresult = std::__parallel::set_intersection(list1, list1+size1, list2, list2+size2, result);
	return endresult-result;
#else
	return 0;
#endif
}

size_t intersect_scalar_branchless_c(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
	size_t counter=0, pos1=0, pos2=0;
	while(pos1 < size1 && pos2 < size2){
		uint32_t data1=list1[pos1], data2=list2[pos2];
		// a bit better, still not nearly as good as asm
		result[counter] = data1;
		counter += (data1 == data2);
		pos1 += (data1 <= data2);
		pos2 += (data1 >= data2);
	}
	return counter;
}
size_t intersect_scalar_branchless_c_count(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2){
	size_t counter=0, pos1=0, pos2=0;
	while(pos1 < size1 && pos2 < size2){
		uint32_t data1=list1[pos1], data2=list2[pos2];
		counter += (data1 == data2);
		pos1 += (data1 <= data2);
		pos2 += (data1 >= data2);
	}
	return counter;
}

size_t intersect_scalar_branchless(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
	uint32_t *end1 = list1+size1, *end2 = list2+size2, *endresult;
	asm(".intel_syntax noprefix;"
		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor rcx, rcx;"
	"1: "
		"cmp %1, %4;"  // list1 != end1
		"je 2f;"
		"cmp %2, %5;"  // list2 != end2
		"je 2f;"

		"mov r10d, [%q2];"  // saved in r10d as value is only 4 byte wide
		"cmp [%q1], r10d;"  // compare *list1 and *list2
		"setle al;"         // set al=1 if lower or equal
		"setge bl;"         // set bl=1 if greater or equal
		"sete  cl;"         // set cl=1 if equal, a bit quicker than: and rax, rbx;

		"mov [%q0], r10d;"  // always save, is overwritten when not equal

		"lea %q1, [%q1 + rax*4];" // list1++, if lower or equal
		"lea %q2, [%q2 + rbx*4];" // list2++, if greater or equal
		"lea %q0, [%q0 + rcx*4];" // result++, if equal

		"jmp 1b;"       // to loop head
	"2: "
		".att_syntax;"

		: "=r"(endresult)
		: "r"(list1), "r"(list2), "0"(result), "r"(end1), "r"(end2)
		: "%rax","%rbx","%rcx", "%r10", "memory", "cc"
	);
	return endresult-result;
}
size_t intersect_scalar_branchless_count(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2){
	uint32_t *end1 = list1+size1, *end2 = list2+size2;
	size_t count=0;
	asm(".intel_syntax noprefix;"
		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor rcx, rcx;"
	"1: "
		"cmp %1, %4;"  // list1 != end1
		"je 2f;"
		"cmp %2, %5;"  // list2 != end2
		"je 2f;"

		"mov r10d, [%q2];"  // saved in r10d as value is only 4 byte wide
		"cmp [%q1], r10d;"  // compare *list1 and *list2
		"setle al;"         // set al=1 if lower or equal
		"setge bl;"         // set bl=1 if greater or equal
		"sete  cl;"         // set cl=1 if equal, a bit quicker than: and rax, rbx;

		"lea %q1, [%q1 + rax*4];" // list1++, if lower or equal
		"lea %q2, [%q2 + rbx*4];" // list2++, if greater or equal
		"lea %q0, [%q0 + rcx];"   // count++, if equal

		"jmp 1b;"       // to loop head
	"2: "
		".att_syntax;"

		: "=r"(count)
		: "r"(list1), "r"(list2), "0"(count), "r"(end1), "r"(end2)
		: "%rax","%rbx","%rcx", "%r10"/*, "memory", "cc"*/
	);
	return count;
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

// taken from: https://highlyscalable.wordpress.com/2012/06/05/fast-intersection-sorted-lists-sse/
size_t intersect_vector_SSE(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
	size_t count = 0;
#ifdef __SSE2__
	size_t i_a = 0, i_b = 0;

	// trim lengths to be a multiple of 4
	size_t st_a = (size1 / 4) * 4;
	size_t st_b = (size2 / 4) * 4;

	while(i_a < st_a && i_b < st_b) {
		//[ load segments of four 32-bit elements
		__m128i v_a = _mm_load_si128((__m128i*)&list1[i_a]);
		__m128i v_b = _mm_load_si128((__m128i*)&list2[i_b]);
		//]

		//[ move pointers
// 		int32_t a_max = _mm_extract_epi32(v_a, 3);
		int32_t a_max = list1[i_a+3];
// 		int32_t b_max = _mm_extract_epi32(v_b, 3);
		int32_t b_max = list2[i_b+3];
		i_a += (a_max <= b_max) * 4;
		i_b += (a_max >= b_max) * 4;
		//]

		// not usable here but mentioned in paper: _mm_slli_si128
		//[ compute mask of common elements
		constexpr int32_t cyclic_shift = _MM_SHUFFLE(0,3,2,1);
		__m128i cmp_mask1 = _mm_cmpeq_epi32(v_a, v_b);    // pairwise comparison
		v_b = _mm_shuffle_epi32(v_b, cyclic_shift);       // shuffling
		__m128i cmp_mask2 = _mm_cmpeq_epi32(v_a, v_b);    // again...
		v_b = _mm_shuffle_epi32(v_b, cyclic_shift);
		__m128i cmp_mask3 = _mm_cmpeq_epi32(v_a, v_b);    // and again...
		v_b = _mm_shuffle_epi32(v_b, cyclic_shift);
		__m128i cmp_mask4 = _mm_cmpeq_epi32(v_a, v_b);    // and again.
		__m128i cmp_mask = _mm_or_si128(
				_mm_or_si128(cmp_mask1, cmp_mask2),
				_mm_or_si128(cmp_mask3, cmp_mask4)
		); // OR-ing of comparison masks
		// convert the 128-bit mask to the 4-bit mask
		int32_t mask = _mm_movemask_ps((__m128)cmp_mask);
		//]

		//[ copy out common elements
		__m128i p = _mm_shuffle_epi8(v_a, shuffle_mask[mask]);
		_mm_storeu_si128((__m128i*)&result[count], p);
		count += _mm_popcnt_u32(mask); // a number of elements is a weight of the mask
		//]
	}

	// intersect the tail using scalar intersection
	count += intersect_scalar(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

#endif
	return count;
}
size_t intersect_vector_SSE_count(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2){
	size_t count = 0;
#ifdef __SSE2__
	size_t i_a = 0, i_b = 0;

	// trim lengths to be a multiple of 4
	size_t st_a = (size1 / 4) * 4;
	size_t st_b = (size2 / 4) * 4;

	while(i_a < st_a && i_b < st_b) {
		//[ load segments of four 32-bit elements
		__m128i v_a = _mm_load_si128((__m128i*)&list1[i_a]);
		__m128i v_b = _mm_load_si128((__m128i*)&list2[i_b]);
		//]

		//[ move pointers
// 		int32_t a_max = _mm_extract_epi32(v_a, 3);
		int32_t a_max = list1[i_a+3];
// 		int32_t b_max = _mm_extract_epi32(v_b, 3);
		int32_t b_max = list2[i_b+3];
		i_a += (a_max <= b_max) * 4;
		i_b += (a_max >= b_max) * 4;
		//]

		// not usable here but mentioned in paper: _mm_slli_si128
		//[ compute mask of common elements
		constexpr int32_t cyclic_shift = _MM_SHUFFLE(0,3,2,1);
		__m128i cmp_mask1 = _mm_cmpeq_epi32(v_a, v_b);    // pairwise comparison
		v_b = _mm_shuffle_epi32(v_b, cyclic_shift);       // shuffling
		__m128i cmp_mask2 = _mm_cmpeq_epi32(v_a, v_b);    // again...
		v_b = _mm_shuffle_epi32(v_b, cyclic_shift);
		__m128i cmp_mask3 = _mm_cmpeq_epi32(v_a, v_b);    // and again...
		v_b = _mm_shuffle_epi32(v_b, cyclic_shift);
		__m128i cmp_mask4 = _mm_cmpeq_epi32(v_a, v_b);    // and again.
		__m128i cmp_mask = _mm_or_si128(
				_mm_or_si128(cmp_mask1, cmp_mask2),
				_mm_or_si128(cmp_mask3, cmp_mask4)
		); // OR-ing of comparison masks
		// convert the 128-bit mask to the 4-bit mask
		int32_t mask = _mm_movemask_ps((__m128)cmp_mask);
		//]

		count += _mm_popcnt_u32(mask); // a number of elements is a weight of the mask
	}

	// intersect the tail using scalar intersection
	count += intersect_scalar_count(list1+i_a, size1-i_a, list2+i_b, size2-i_b);

#endif
	return count;
}

size_t intersect_vector_SSE_asm(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
	size_t count = 0;
#ifdef __SSE2__
	size_t i_a = 0, i_b = 0;

	// trim lengths to be a multiple of 4
	size_t st_a = (size1 / 4) * 4;
	size_t st_b = (size2 / 4) * 4;

	while(i_a < st_a && i_b < st_b) {
		asm(".intel_syntax noprefix;"

			"vmovdqa xmm0, [%q3 + %q0*4];" //__m128i v_a = _mm_load_si128((__m128i*)&list1[i_a]);
			"vmovdqa xmm1, [%q4 + %q1*4];" //__m128i v_b = _mm_load_si128((__m128i*)&list2[i_b]);

			"mov r8d, [%q3 + %q0*4 + 12];" //int32_t a_max = list1[i_a+3];
			"mov r9d, [%q4 + %q1*4 + 12];" //int32_t b_max = list2[i_b+3];
			//i_a += (a_max <= b_max) * 4;
			//i_b += (a_max >= b_max) * 4;
			"xor rax, rax;"
			"xor rbx, rbx;"
			"cmp r8d, r9d;"
			"setle al;"
			"setge bl;"
			"lea %q0, [%q0 + rax*4];"
			"lea %q1, [%q1 + rbx*4];"

// 			"vpcmpeqd xmm2, xmm0, xmm1;"
// 			"vpshufd xmm1, xmm1, 0x39;"
// 			"vpcmpeqd xmm3, xmm0, xmm1;"
// 			"vpshufd xmm1, xmm1, 0x39;"
// 			"vpcmpeqd xmm4, xmm0, xmm1;"
// 			"vpshufd xmm1, xmm1, 0x39;"
// 			"vpcmpeqd xmm5, xmm0, xmm1;"
			"vpshufd xmm6, xmm1, 0x39;"
			"vpshufd xmm7, xmm6, 0x39;"
			"vpshufd xmm8, xmm7, 0x39;"
			"vpcmpeqd xmm2, xmm0, xmm1;"
			"vpcmpeqd xmm3, xmm0, xmm6;"
			"vpcmpeqd xmm4, xmm0, xmm7;"
			"vpcmpeqd xmm5, xmm0, xmm8;"

			"vpor xmm2, xmm2, xmm3;"
			"vpor xmm4, xmm4, xmm5;"
			"vpor xmm2, xmm2, xmm4;"

			"vmovmskps r8d, xmm2;"
			// save in result
			// 16 multiplier not possible in index expression
			// -> do it outside
			"movslq r9, r8d;"
			"shl r9, 4;"
			"vpshufb xmm0, xmm0, [%q6 + r9];"
			"vmovups [%q5 + %q2*4], xmm0;"

			"popcnt r8d, r8d;"
// 			"movl r8, r8d;" // zero extend
			"add %q2, r8;"

			".att_syntax;"
			: "=r"(i_a), "=r"(i_b), "=r"(count)
			: "r"(list1), "r"(list2), "r"(result), "r"(shuffle_mask),
				"0"(i_a), "1"(i_b), "2"(count)
			: "%rax", "%rbx", "%r8", "%r9",
				"%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5",
				"%xmm6", "%xmm7", "xmm8",
				"memory", "cc"
		);
	}
	// intersect the tail using scalar intersection
	count += intersect_scalar(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

#endif
	return count;
}

static uint32_t *shuffle_mask_avx;
void prepare_shuffling_dictionary_avx(){
	shuffle_mask_avx = (uint32_t*)aligned_alloc(32, 256*8*sizeof(uint32_t));
	for(uint32_t i=0; i<256; ++i){
		int count=0, rest=7;
		for(int b=0; b<8; ++b){
			if(i & (1 << b)){
				// n index at pos p - move nth element to pos p
				shuffle_mask_avx[i*8 + count] = b; // move all set bits to beginning
				++count;
			}else{
				shuffle_mask_avx[i*8 + rest] = b; // move rest at the end
				--rest;
			}
		}
	}
}
#ifdef __AVX__
// taken from asmlib by agner: http://www.agner.org/optimize/
// emulate permute8x32 AVX2-instruction in AVX
inline __m256 permute8x32(__m256 const &table, __m256i const &index){
	// swap low and high part of table
// 	__m256  t1 = _mm256_castps128_ps256(_mm256_extractf128_ps(table, 1));
// 	__m256  t2 = _mm256_insertf128_ps(t1, _mm256_castps256_ps128(table), 1);
	__m256  t2 = _mm256_permute2f128_ps(table, table, 1);
	// join index parts
// 	__m256i index2 = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm256_castpd256_pd128(index)), _mm256_extractf128_pd(index, 1), 1);
	// permute within each 128-bit part
	__m256  r0 = _mm256_permutevar_ps(table, index);
	__m256  r1 = _mm256_permutevar_ps(t2,    index);
	// high index bit for blend
// 	__m128i k1 = _mm_slli_epi32(_mm256_extractf128_si256(index, 1) ^ 4, 29); // constant for xor is buggy, generates {4,0,4,0}
	__m128i k1 = _mm_slli_epi32(
		_mm_xor_si128(
			_mm256_extractf128_si256(index, 1),
			_mm_set1_epi32(4)
		), 29);
	__m128i k0 = _mm_slli_epi32(_mm256_castsi256_si128(index),       29);
	__m256  kk = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(k0)), _mm_castsi128_ps(k1), 1);
	// blend the two permutes
	return _mm256_blendv_ps(r0, r1, kk);
}
#endif

size_t intersect_vector_avx(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
#ifdef __AVX__
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;
	while(i_a < st_a && i_b < st_b){
		__m256i v_a = _mm256_load_si256((__m256i*)&list1[i_a]);
		__m256i v_b = _mm256_load_si256((__m256i*)&list2[i_b]);

// 		int32_t a_max = _mm256_extract_epi32(v_a, 7);
		int32_t a_max = list1[i_a+7];
// 		int32_t b_max = _mm256_extract_epi32(v_b, 7);
		int32_t b_max = list2[i_b+7];
		i_a += (a_max <= b_max) * 8;
		i_b += (a_max >= b_max) * 8;

		// AVX is missing many integer operations, AVX2 has them
		__m256 vfa = _mm256_cvtepi32_ps(v_a);
		__m256 vfb = _mm256_cvtepi32_ps(v_b);

		constexpr int32_t cyclic_shift = _MM_SHUFFLE(0,3,2,1); //rotating right
		constexpr int32_t cyclic_shift2= _MM_SHUFFLE(2,1,0,3); //rotating left
		constexpr int32_t cyclic_shift3= _MM_SHUFFLE(1,0,3,2); //between
		// AVX2: _mm256_cmpeq_epi32
		__m256 cmp_mask1 = _mm256_cmp_ps(vfa, vfb, _CMP_EQ_OQ);
		__m256 rot1 = _mm256_permute_ps(vfb, cyclic_shift);
		__m256 cmp_mask2 = _mm256_cmp_ps(vfa, rot1, _CMP_EQ_OQ);
		__m256 rot2 = _mm256_permute_ps(vfb, cyclic_shift3);
		__m256 cmp_mask3 = _mm256_cmp_ps(vfa, rot2, _CMP_EQ_OQ);
		__m256 rot3 = _mm256_permute_ps(vfb, cyclic_shift2);
		__m256 cmp_mask4 = _mm256_cmp_ps(vfa, rot3, _CMP_EQ_OQ);

		__m256 rot4 = _mm256_permute2f128_ps(vfb, vfb, 1);

		__m256 cmp_mask5 = _mm256_cmp_ps(vfa, rot4, _CMP_EQ_OQ);
		__m256 rot5 = _mm256_permute_ps(rot4, cyclic_shift);
		__m256 cmp_mask6 = _mm256_cmp_ps(vfa, rot5, _CMP_EQ_OQ);
		__m256 rot6 = _mm256_permute_ps(rot4, cyclic_shift3);
		__m256 cmp_mask7 = _mm256_cmp_ps(vfa, rot6, _CMP_EQ_OQ);
		__m256 rot7 = _mm256_permute_ps(rot4, cyclic_shift2);
		__m256 cmp_mask8 = _mm256_cmp_ps(vfa, rot7, _CMP_EQ_OQ);

		// AVX2: _mm256_or_si256
		__m256 cmp_mask = _mm256_or_ps(
			_mm256_or_ps(
				_mm256_or_ps(cmp_mask1, cmp_mask2),
				_mm256_or_ps(cmp_mask3, cmp_mask4)
			),
			_mm256_or_ps(
				_mm256_or_ps(cmp_mask5, cmp_mask6),
				_mm256_or_ps(cmp_mask7, cmp_mask8)
			)
		);
		int32_t mask = _mm256_movemask_ps(cmp_mask);
		// just use unchanged v_a, don't convert back vfa
		// AVX2: _mm256_maskstore_epi32 directly with cmp_mask
		// just use float variant
// 		_mm256_maskstore_ps((float*)&result[count], (__m256i)cmp_mask, (__m256)v_a);
// 		_mm256_store_si256();
		__m256i idx = _mm256_load_si256((const __m256i*)&shuffle_mask_avx[mask*8]);
		__m256i p = (__m256i)permute8x32((__m256)v_a, idx);
		_mm256_storeu_si256((__m256i*)&result[count], p);

		count += _mm_popcnt_u32(mask);
	}
	// intersect the tail using scalar intersection
	count += intersect_scalar(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

#endif
	return count;
}
size_t intersect_vector_avx_count(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2){
	size_t count=0, i_a=0, i_b=0;
#ifdef __AVX__
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;
	while(i_a < st_a && i_b < st_b){
		__m256i v_a = _mm256_load_si256((__m256i*)&list1[i_a]);
		__m256i v_b = _mm256_load_si256((__m256i*)&list2[i_b]);

// 		int32_t a_max = _mm256_extract_epi32(v_a, 7);
		uint32_t a_max = list1[i_a+7];
// 		int32_t b_max = _mm256_extract_epi32(v_b, 7);
		uint32_t b_max = list2[i_b+7];
		i_a += (a_max <= b_max) * 8;
		i_b += (a_max >= b_max) * 8;

		// AVX is missing many integer operations, AVX2 has them
		__m256 vfa = _mm256_cvtepi32_ps(v_a);
		__m256 vfb = _mm256_cvtepi32_ps(v_b);

		constexpr int32_t cyclic_shift = _MM_SHUFFLE(0,3,2,1); //rotating right
		constexpr int32_t cyclic_shift2= _MM_SHUFFLE(2,1,0,3); //rotating left
		constexpr int32_t cyclic_shift3= _MM_SHUFFLE(1,0,3,2); //between
		// AVX2: _mm256_cmpeq_epi32
		__m256 cmp_mask1 = _mm256_cmp_ps(vfa, vfb, _CMP_EQ_OQ);
		__m256 rot1 = _mm256_permute_ps(vfb, cyclic_shift);
		__m256 cmp_mask2 = _mm256_cmp_ps(vfa, rot1, _CMP_EQ_OQ);
		__m256 rot2 = _mm256_permute_ps(vfb, cyclic_shift3);
		__m256 cmp_mask3 = _mm256_cmp_ps(vfa, rot2, _CMP_EQ_OQ);
		__m256 rot3 = _mm256_permute_ps(vfb, cyclic_shift2);
		__m256 cmp_mask4 = _mm256_cmp_ps(vfa, rot3, _CMP_EQ_OQ);

		__m256 rot4 = _mm256_permute2f128_ps(vfb, vfb, 1);

		__m256 cmp_mask5 = _mm256_cmp_ps(vfa, rot4, _CMP_EQ_OQ);
		__m256 rot5 = _mm256_permute_ps(rot4, cyclic_shift);
		__m256 cmp_mask6 = _mm256_cmp_ps(vfa, rot5, _CMP_EQ_OQ);
		__m256 rot6 = _mm256_permute_ps(rot4, cyclic_shift3);
		__m256 cmp_mask7 = _mm256_cmp_ps(vfa, rot6, _CMP_EQ_OQ);
		__m256 rot7 = _mm256_permute_ps(rot4, cyclic_shift2);
		__m256 cmp_mask8 = _mm256_cmp_ps(vfa, rot7, _CMP_EQ_OQ);

		// AVX2: _mm256_or_si256
		__m256 cmp_mask = _mm256_or_ps(
			_mm256_or_ps(
				_mm256_or_ps(cmp_mask1, cmp_mask2),
				_mm256_or_ps(cmp_mask3, cmp_mask4)
			),
			_mm256_or_ps(
				_mm256_or_ps(cmp_mask5, cmp_mask6),
				_mm256_or_ps(cmp_mask7, cmp_mask8)
			)
		);
		uint32_t mask = _mm256_movemask_ps(cmp_mask);
		count += _mm_popcnt_u32(mask);
	}
	// intersect the tail using scalar intersection
	count += intersect_scalar_count(list1+i_a, size1-i_a, list2+i_b, size2-i_b);

#endif
	return count;
}

//FIXME: broken atm
size_t intersect_vector_avx_asm(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
#ifdef __AVX__
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;
	uint32_t xorconst[] = {4,4,4,4};

	asm(".intel_syntax noprefix;"

		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor r9, r9;"
		"vmovdqa xmm15, [%q10];"
	"1: "
		"cmp %1, %4;"
		"je 2f;"
		"cmp %2, %5;"
		"je 2f;"

		"vmovdqa ymm0, [%q6 + %q1*4];" // elements are 4 byte
		"vcvtdq2ps ymm1, ymm0;"
		"vcvtdq2ps ymm2, [%q7 + %q2*4];"

		"mov r8d, [%q6 + %q1*4 + 28];" //int32_t a_max = list1[i_a+7];
		"cmp r8d, [%q7 + %q2*4 + 28];"
		"setle al;"
		"setge bl;"
		"lea %q1, [%q1 + rax*8];"
		"lea %q2, [%q2 + rbx*8];"

		"vcmpeqps ymm10, ymm1, ymm2;"
		"vpermilps ymm3, ymm2, 0x39;"
		"vcmpeqps ymm11, ymm1, ymm3;"
		"vpermilps ymm4, ymm3, 0x39;"
		"vcmpeqps ymm4, ymm1, ymm4;"
		"vpermilps ymm5, ymm2, 0x93;"
		"vcmpeqps ymm5, ymm1, ymm5;"

		"vperm2f128 ymm6, ymm2, ymm2, 1;"
		"vcmpeqps ymm12, ymm1, ymm6;"

		"vpermilps ymm7, ymm6, 0x39;"
		"vcmpeqps ymm13, ymm1, ymm7;"
		"vpermilps ymm8, ymm7, 0x39;"
		"vcmpeqps ymm8, ymm1, ymm8;"
		"vpermilps ymm9, ymm6, 0x93;"
		"vcmpeqps ymm9, ymm1, ymm9;"

		"vorps ymm10, ymm10, ymm11;"
		"vorps ymm4, ymm4, ymm5;"
		"vorps ymm12, ymm12, ymm13;"
		"vorps ymm8, ymm8, ymm9;"

		"vorps ymm10, ymm10, ymm4;"
		"vorps ymm12, ymm12, ymm8;"

		"vorps ymm10, ymm10, ymm12;"

		"vmovmskps r9d, ymm10;"

// 		"vmaskmovps [%q8 + %q0*4], ymm10, ymm0;"
		"lea r10, [r9*8];"
		"vmovdqa ymm14, [%q9 + r10*4];" // shuffle_mask_avx
		// permute8x32
		"vperm2f128 ymm7, ymm0, ymm0, 1;" // swap table
		"vpermilps ymm0, ymm0, ymm14;"
		"vpermilps ymm7, ymm7, ymm14;"
		"vextractf128 xmm2, ymm14, 0x1;"
		"vpxor xmm2, xmm2, xmm15;"
		"vpslld xmm2, xmm2, 0x1d;"   //k1
		"vpslld xmm14, xmm14, 0x1d;" //k0
		"vinsertf128 ymm14, ymm14, xmm2, 0x1;"
		"vblendvps ymm0, ymm0, ymm7, ymm14;"
		"vmovdqu [%q8 + %q0*4], ymm0;"

		"popcnt r9d, r9d;"
		"add %q0, r9;"

		"jmp 1b;"
	"2: "
		".att_syntax;"
		: "=r"(count), "=r"(i_a), "=r"(i_b)
		: "0"(count), "r"(st_a), "r"(st_b),
			"r"(list1), "r"(list2), "r"(result),
			"r"(shuffle_mask_avx),/*10*/"r"(xorconst),
			"1"(i_a), "2"(i_b)
		: "%rax", "%rbx", "%r8", "%r9","%r10",
			"ymm0","ymm1","ymm2","ymm3","ymm4",
			"ymm5","ymm6","ymm7","ymm8","ymm9",
			"ymm10","ymm11","ymm12","ymm13","ymm14","ymm15",
			"memory", "cc"
	);
	// intersect the tail using scalar intersection
	count += intersect_scalar_branchless(
		list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count
	);
#endif
	return count;
}
size_t intersect_vector_avx_asm_count(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2){
	size_t count=0, i_a=0, i_b=0;
#ifdef __AVX__
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;

	asm(".intel_syntax noprefix;"

		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor r9, r9;"
	"1: "
// 		"cmp %1, %4;"
// 		"je 2f;"
		"cmp %2, %5;"
		"je 2f;"

		"vcvtdq2ps ymm1, [%q6 + %q1*4];" // elements are 4 byte
		"vcvtdq2ps ymm2, [%q7 + %q2*4];"

		"mov r8d, [%q6 + %q1*4 + 28];" //int32_t a_max = list1[i_a+7];
		"cmp r8d, [%q7 + %q2*4 + 28];"
		"setle al;"
		"setge bl;"
		"lea %q1, [%q1 + rax*8];"
		"lea %q2, [%q2 + rbx*8];"

		"vcmpeqps ymm10, ymm1, ymm2;"
		"vperm2f128 ymm6, ymm2, ymm2, 1;"
		"vpermilps ymm3, ymm2, 0x39;"
		"vpermilps ymm4, ymm2, 0x4e;"
		"vcmpeqps ymm11, ymm1, ymm3;"
		"vpermilps ymm5, ymm2, 0x93;"
		"vcmpeqps ymm4, ymm1, ymm4;"
		"vpermilps ymm7, ymm6, 0x39;"
		"vpermilps ymm8, ymm6, 0x4e;"
		"vpermilps ymm9, ymm6, 0x93;"
		"vcmpeqps ymm5, ymm1, ymm5;"

		"vorps ymm10, ymm10, ymm11;"
		"vcmpeqps ymm12, ymm1, ymm6;"
		"vorps ymm4, ymm4, ymm5;"

		"vcmpeqps ymm13, ymm1, ymm7;"
		"vcmpeqps ymm8, ymm1, ymm8;"
		"vcmpeqps ymm9, ymm1, ymm9;"

		"vorps ymm12, ymm12, ymm13;"
		"vorps ymm8, ymm8, ymm9;"

		"vorps ymm10, ymm10, ymm4;"
		"vorps ymm12, ymm12, ymm8;"

		"vorps ymm10, ymm10, ymm12;"

		"vmovmskps r9d, ymm10;"

		"popcnt r9d, r9d;"
		"add %q0, r9;"

// 		"jmp 1b;"
		"cmp %1, %4;"
		"jb 1b;"
	"2: "
		".att_syntax;"
		: "=r"(count), "=r"(i_a), "=r"(i_b)
		: "0"(count), "r"(st_a), "r"(st_b),
			"r"(list1), "r"(list2),
			"1"(i_a), "2"(i_b)
		: "%rax", "%rbx", "%r8", "%r9","%r10",
			"ymm0","ymm1","ymm2","ymm3","ymm4",
			"ymm5","ymm6","ymm7","ymm8","ymm9",
			"ymm10","ymm11","ymm12","ymm13","ymm14","ymm15",
			"memory", "cc"
	);
	// intersect the tail using scalar intersection
	count += intersect_scalar_branchless_count(
		list1+i_a, size1-i_a, list2+i_b, size2-i_b
	);
#endif
	return count;
}

// a naive version inspired by this paper:
// "SIMD Compression and the Intersection of Sorted Integers"
// source code: https://github.com/lemire/SIMDCompressionAndIntersection
size_t intersect_galloping_V1_AVX(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
	size_t i=0, j=0, count=0;
#ifdef __AVX__

	for(; i<size1; ++i){
		uint32_t ri = list1[i];
		__m256i R = _mm256_set1_epi32(ri);

		while(list2[j+15] < ri){
			j += 16;
			if(j > size2){
				j -= 16; //FIXME
				goto finishscalar;
			}
		}
		__m256i F = _mm256_load_si256((__m256i*)&list2[j]);
		__m256i F2 = _mm256_load_si256((__m256i*)&list2[j+8]);

// 		_mm256_cmpeq_epi32(); //AVX2
		__m256 vfr = _mm256_cvtepi32_ps(R);
		__m256 vff = _mm256_cvtepi32_ps(F);
		__m256 vff2 = _mm256_cvtepi32_ps(F2);

		__m256 cmp = _mm256_cmp_ps(vfr, vff, _CMP_EQ_OQ);
		__m256 cmp2 = _mm256_cmp_ps(vfr, vff2, _CMP_EQ_OQ);
		cmp = _mm256_or_ps(cmp, cmp2);

		result[count] = ri;
		asm(".intel_syntax noprefix;"

			"xor rax, rax;"
			"vtestps %1, %1;" //_mm256_testz_ps(cmp, cmp)
			"setnz al;"
			"add %q0, rax;"   // count++, if cmp non-zero

			".att_syntax;"
			: "=r"(count)
			: "x"(cmp), "0"(count)
			: "%eax"
		);
	}
finishscalar:
	// intersect the tail using scalar intersection
	while(i < size1 && j < size2){
		uint32_t data1=list1[i], data2=list2[j];
		count += (data1 == data2);
		i += (data1 <= data2);
		j += (data1 >= data2);
	}
#endif
	return count;
}
size_t intersect_galloping_V1_AVX_count(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2){
	size_t i=0, j=0, count=0;
#ifdef __AVX__
	size_t st_b = (size2 / 16) * 16;

	for(; i<size1; ++i){
		uint32_t ri = list1[i];
		__m256i R = _mm256_set1_epi32(ri);

		while(list2[j+15] < ri){
			j += 16;
			if(j >= st_b){
				j -= 16; //FIXME
				goto finishscalar;
			}
		}
		__m256i F = _mm256_load_si256((__m256i*)&list2[j]);
		__m256i F2 = _mm256_load_si256((__m256i*)&list2[j+8]);

// 		_mm256_cmpeq_epi32(); //AVX2
		__m256 vfr = _mm256_cvtepi32_ps(R);
		__m256 vff = _mm256_cvtepi32_ps(F);
		__m256 vff2 = _mm256_cvtepi32_ps(F2);

		__m256 cmp = _mm256_cmp_ps(vfr, vff, _CMP_EQ_OQ);
		__m256 cmp2 = _mm256_cmp_ps(vfr, vff2, _CMP_EQ_OQ);
		cmp = _mm256_or_ps(cmp, cmp2);

// 		result[count] = ri;
		asm(".intel_syntax noprefix;"

			"xor rax, rax;"
			"vtestps %1, %1;" //_mm256_testz_ps(cmp, cmp)
			"setnz al;"
			"add %q0, rax;"   // count++, if cmp non-zero

			".att_syntax;"
			: "=r"(count)
			: "x"(cmp), "0"(count)
			: "%eax"
		);
	}
finishscalar:
	// intersect the tail using scalar intersection
	while(i < size1 && j < size2){
		uint32_t data1=list1[i], data2=list2[j];
		count += (data1 == data2);
		i += (data1 <= data2);
		j += (data1 >= data2);
	}
#endif
	return count;
}

size_t intersect_galloping_V1_SSE(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
	size_t i=0, j=0, count=0;
#ifdef __SSE4_2__

	for(; i<size1; ++i){
		uint32_t ri = list1[i];
		__m128i R = _mm_set1_epi32(ri);

		while(list2[j+7] < ri){
			j += 8;
			if(j > size2){
				j -= 8; //FIXME
				goto finishscalar;
			}
		}
		__m128i F = _mm_load_si128((__m128i*)&list2[j]);
		__m128i F2= _mm_load_si128((__m128i*)&list2[j+4]);

		__m128i cmp = _mm_cmpeq_epi32(R, F);
		__m128i cmp2 = _mm_cmpeq_epi32(R, F2);
		cmp = _mm_or_si128(cmp, cmp2);

		result[count] = ri;
		asm(".intel_syntax noprefix;"

			"xor rax, rax;"
			"ptest %1, %1;"
			"setnz al;"
			"add %q0, rax;"   // count++, if cmp non-zero

			".att_syntax;"
			: "=r"(count)
			: "x"(cmp), "0"(count)
			: "%eax"
		);
	}
finishscalar:
	// intersect the tail using scalar intersection
	while(i < size1 && j < size2){
		uint32_t data1=list1[i], data2=list2[j];
		count += (data1 == data2);
		i += (data1 <= data2);
		j += (data1 >= data2);
	}
#endif
	return count;
}
size_t intersect_galloping_V1_SSE_count(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2){
	size_t i=0, j=0, count=0;
#ifdef __SSE4_2__

	for(; i<size1; ++i){
		uint32_t ri = list1[i];
		__m128i R = _mm_set1_epi32(ri);

		while(list2[j+7] < ri){
			j += 8;
			if(j > size2){
				j -= 8; //FIXME
				goto finishscalar;
			}
		}
		__m128i F = _mm_load_si128((__m128i*)&list2[j]);
		__m128i F2= _mm_load_si128((__m128i*)&list2[j+4]);

		__m128i cmp = _mm_cmpeq_epi32(R, F);
		__m128i cmp2 = _mm_cmpeq_epi32(R, F2);
		cmp = _mm_or_si128(cmp, cmp2);

		asm(".intel_syntax noprefix;"

			"xor rax, rax;"
			"ptest %1, %1;"
			"setnz al;"
			"add %q0, rax;"   // count++, if cmp non-zero

			".att_syntax;"
			: "=r"(count)
			: "x"(cmp), "0"(count)
			: "%eax"
		);
	}
finishscalar:
	// intersect the tail using scalar intersection
	while(i < size1 && j < size2){
		uint32_t data1=list1[i], data2=list2[j];
		count += (data1 == data2);
		i += (data1 <= data2);
		j += (data1 >= data2);
	}
#endif
	return count;
}


void run(uint32_t **lists,
	size_t (*func)(uint32_t*,size_t,uint32_t*,size_t,uint32_t*)=nullptr,
	size_t (*func_count)(uint32_t*,size_t,uint32_t*,size_t)=nullptr
){
#if 1
	if(func){
		auto t_start = std::chrono::high_resolution_clock::now();
		size_t intersected=0;
		for(size_t repeat=0; repeat<repeatCount; ++repeat){
			size_t i;
			for(/*size_t */i=0; i<listCount; ++i){
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
#else
	constexpr size_t size = 20;
	for(size_t i=0; i<size; ++i){
		printf("%3i; %3i\n", lists[0][i], lists[1][i]);
	}

	uint32_t *intersected_list = (uint32_t*)aligned_alloc(32, arraySize*sizeof(uint32_t));
	size_t intersected = func(lists[0], size, lists[1], size, intersected_list);

	puts("intersected:");
	for(size_t i=0; i<intersected; ++i){
		printf("%i, ", intersected_list[i]);
	}
	printf("\n\n");
	free(intersected_list);
#endif
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
	std::cout << "preparing lists done - "
		<< std::chrono::duration<double, std::milli>(t_end-t_start).count()<< " ms"
		<< std::endl;

	puts("scalar:");
	run(lists, intersect_scalar, intersect_scalar_count);
	puts("stl set_intersection:");
	run(lists, intersect_scalar_stl);
	puts("stl parallel set_intersection: uses more than one core, just for reference here");
	run(lists, intersect_scalar_stl_parallel);
	puts("c branchless scalar:");
	run(lists, intersect_scalar_branchless_c, intersect_scalar_branchless_c_count);
	puts("asm branchless scalar:");
	run(lists, intersect_scalar_branchless, intersect_scalar_branchless_count);

	prepare_shuffling_dictionary();
	puts("128bit SSE vector:");
	run(lists, intersect_vector_SSE, intersect_vector_SSE_count);
	puts("SSE asm:");
	run(lists, intersect_vector_SSE_asm);

	prepare_shuffling_dictionary_avx();
	puts("256bit AVX vector: (not AVX2)");
	run(lists, intersect_vector_avx, intersect_vector_avx_count);
	puts("256bit AVX vector: (not AVX2) - asm");
	run(lists, /*intersect_vector_avx_asm*/nullptr, intersect_vector_avx_asm_count); //FIXME: normal intersection segfaults
	free(shuffle_mask_avx);

	puts("SIMD Galloping V1: AVX");
	run(lists, intersect_galloping_V1_AVX, intersect_galloping_V1_AVX_count);
// 	puts("SIMD Galloping V1: SSE4.2"); //FIXME: broken
// 	run(lists, intersect_galloping_V1_SSE, intersect_galloping_V1_SSE_count);

	for(size_t i=0; i<listCount; ++i){
		free(lists[i]);
	}
	delete[] lists;

	return 0;
}
