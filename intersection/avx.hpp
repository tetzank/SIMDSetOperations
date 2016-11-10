#ifndef INTERSECTION_AVX_HPP_
#define INTERSECTION_AVX_HPP_

#include <immintrin.h>

#include "branchless.hpp"


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

size_t intersect_vector_avx(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
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
size_t intersect_vector_avx_count(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2){
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
#if 0
size_t intersect_vector_avx_asm(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
#ifdef __AVX__
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;
	uint32_t xorconst[] = {4,4,4,4};

	asm(".intel_syntax noprefix;"

		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor r9, r9;"
		"vmovdqa xmm15, [%q[xorconst]];"
	"1: "
		"cmp %[i_a], %[st_a];"
		"je 2f;"
		"cmp %[i_b], %[st_b];"
		"je 2f;"

		"vmovdqa ymm0, [%q[list1] + %q[i_a]*4];" // elements are 4 byte
		"vcvtdq2ps ymm1, ymm0;"
		"vcvtdq2ps ymm2, [%q[list2] + %q[i_b]*4];"

		"mov r8d, [%q[list1] + %q[i_a]*4 + 28];" //int32_t a_max = list1[i_a+7];
		"cmp r8d, [%q[list2] + %q[i_b]*4 + 28];"
		"setle al;"
		"setge bl;"
		"lea %q[i_a], [%q[i_a] + rax*8];"
		"lea %q[i_b], [%q[i_b] + rbx*8];"

		"vcmpeqps ymm10, ymm1, ymm2;"
		"vpermilps ymm3, ymm2, 0x39;"
		"vcmpeqps ymm11, ymm1, ymm3;"
		"vpermilps ymm4, ymm3, 0x39;" //FIXME: why two cyclic shifts after eachother? break dependency
		"vcmpeqps ymm4, ymm1, ymm4;"
		"vpermilps ymm5, ymm2, 0x93;"
		"vcmpeqps ymm5, ymm1, ymm5;"

		"vperm2f128 ymm6, ymm2, ymm2, 1;"
		"vcmpeqps ymm12, ymm1, ymm6;"

		"vpermilps ymm7, ymm6, 0x39;"
		"vcmpeqps ymm13, ymm1, ymm7;"
		"vpermilps ymm8, ymm7, 0x39;" //FIXME
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
		"vmovdqa ymm14, [%q[shuffle_mask_avx] + r10*4];" // shuffle_mask_avx
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
		"vmovdqu [%q8 + %q[count]*4], ymm0;"

		"popcnt r9d, r9d;"
		"add %q[count], r9;"

		"jmp 1b;"
	"2: "
		".att_syntax;"
		: [count]"=r"(count), [i_a]"=r"(i_a), [i_b]"=r"(i_b)
		: [st_a]"r"(st_a), [st_b]"r"(st_b),
			[list1]"r"(list1), [list2]"r"(list2), [result]"r"(result),
			[shuffle_mask_avx]"r"(shuffle_mask_avx), [xorconst]"r"(xorconst),
			"0"(count), "1"(i_a), "2"(i_b)
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
#endif
size_t intersect_vector_avx_asm_count(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2){
	size_t count=0, i_a=0, i_b=0;
#ifdef __AVX__
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;

	asm(".intel_syntax noprefix;"

		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor r9, r9;"
	"1: "
// 		"cmp %[i_a], %[st_a];"
// 		"je 2f;"
		"cmp %[i_b], %[st_b];"
		"je 2f;"

		"vcvtdq2ps ymm1, [%q[list1] + %q[i_a]*4];" // elements are 4 byte
		"vcvtdq2ps ymm2, [%q[list2] + %q[i_b]*4];"

		"mov r8d, [%q[list1] + %q[i_a]*4 + 28];" //int32_t a_max = list1[i_a+7];
		"cmp r8d, [%q[list2] + %q[i_b]*4 + 28];"
		"setle al;"
		"setge bl;"
		"lea %q[i_a], [%q[i_a] + rax*8];"
		"lea %q[i_b], [%q[i_b] + rbx*8];"

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
		"add %q[count], r9;"

// 		"jmp 1b;"
		"cmp %[i_a], %[st_a];"
		"jb 1b;"
	"2: "
		".att_syntax;"
		: [count]"=r"(count), [i_a]"=r"(i_a), [i_b]"=r"(i_b)
		: [st_a]"r"(st_a), [st_b]"r"(st_b),
			[list1]"r"(list1), [list2]"r"(list2),
			"0"(count), "1"(i_a), "2"(i_b)
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


#endif
