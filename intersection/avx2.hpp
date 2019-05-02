#ifndef INTERSECTION_AVX2_HPP_
#define INTERSECTION_AVX2_HPP_

#include <immintrin.h>


size_t intersect_vector_avx2(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
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

		__m256i cmp_mask = _mm256_or_si256(
			_mm256_or_si256(
				_mm256_or_si256(cmp_mask1, cmp_mask2),
				_mm256_or_si256(cmp_mask3, cmp_mask4)
			),
			_mm256_or_si256(
				_mm256_or_si256(cmp_mask5, cmp_mask6),
				_mm256_or_si256(cmp_mask7, cmp_mask8)
			)
		);
		int32_t mask = _mm256_movemask_ps((__m256)cmp_mask);

		__m256i idx = _mm256_load_si256((const __m256i*)&shuffle_mask_avx[mask*8]);
		__m256i p = _mm256_permutevar8x32_epi32(v_a, idx);
		_mm256_storeu_si256((__m256i*)&result[count], p);

		count += _mm_popcnt_u32(mask);
	}
	// intersect the tail using scalar intersection
	count += intersect_scalar(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

	return count;
}
size_t intersect_vector_avx2_count(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2){
	size_t count=0, i_a=0, i_b=0;
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

		constexpr int32_t cyclic_shift = _MM_SHUFFLE(0,3,2,1); //rotating right
		constexpr int32_t cyclic_shift2= _MM_SHUFFLE(2,1,0,3); //rotating left
		constexpr int32_t cyclic_shift3= _MM_SHUFFLE(1,0,3,2); //between
		// AVX2: _mm256_cmpeq_epi32
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

		// AVX2: _mm256_or_si256
		__m256i cmp_mask = _mm256_or_si256(
			_mm256_or_si256(
				_mm256_or_si256(cmp_mask1, cmp_mask2),
				_mm256_or_si256(cmp_mask3, cmp_mask4)
			),
			_mm256_or_si256(
				_mm256_or_si256(cmp_mask5, cmp_mask6),
				_mm256_or_si256(cmp_mask7, cmp_mask8)
			)
		);
		int32_t mask = _mm256_movemask_ps((__m256)cmp_mask);
		count += _mm_popcnt_u32(mask);
	}
	// intersect the tail using scalar intersection
	count += intersect_scalar_count(list1+i_a, size1-i_a, list2+i_b, size2-i_b);

	return count;
}

size_t intersect_vector_avx2_asm(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;

	asm(".intel_syntax noprefix;"

		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor r9, r9;"
#if IACA_INTERSECT_AVX2
		IACA_START_ASM
#endif
	"1: "
 		"cmp %[i_a], %[st_a];"
 		"je 2f;"
		"cmp %[i_b], %[st_b];"
		"je 2f;"

		"vmovdqa ymm1, [%q[list1] + %q[i_a]*4];" // elements are 4 byte
		"vmovdqa ymm2, [%q[list2] + %q[i_b]*4];"

		"mov r8d, [%q[list1] + %q[i_a]*4 + 28];" //int32_t a_max = list1[i_a+7];
		"cmp r8d, [%q[list2] + %q[i_b]*4 + 28];"
		"setbe al;"
		"setae bl;"
		"lea %q[i_a], [%q[i_a] + rax*8];"
		"lea %q[i_b], [%q[i_b] + rbx*8];"

		"vpcmpeqd ymm10, ymm1, ymm2;"
		"vperm2f128 ymm6, ymm2, ymm2, 1;"
		"vpermilps ymm3, ymm2, 0x39;"
		"vpermilps ymm4, ymm2, 0x4e;"
		"vpcmpeqd ymm11, ymm1, ymm3;"
		"vpermilps ymm5, ymm2, 0x93;"
		"vpcmpeqd ymm4, ymm1, ymm4;"
		"vpermilps ymm7, ymm6, 0x39;"
		"vpermilps ymm8, ymm6, 0x4e;"
		"vpermilps ymm9, ymm6, 0x93;"
		"vpcmpeqd ymm5, ymm1, ymm5;"

		"vpor ymm10, ymm10, ymm11;"
		"vpcmpeqd ymm12, ymm1, ymm6;"
		"vpor ymm4, ymm4, ymm5;"

		"vpcmpeqd ymm13, ymm1, ymm7;"
		"vpcmpeqd ymm8, ymm1, ymm8;"
		"vpcmpeqd ymm9, ymm1, ymm9;"

		"vpor ymm12, ymm12, ymm13;"
		"vpor ymm8, ymm8, ymm9;"

		"vpor ymm10, ymm10, ymm4;"
		"vpor ymm12, ymm12, ymm8;"

		"vpor ymm10, ymm10, ymm12;"

		"vmovmskps r9d, ymm10;"

		//4 * 8 * r9 => 5x shift
		//"shlx r8, r9, ;" //no immediate, need register for constant 5
		"movsxd r8, r9d;"
		"shl r8, 5;"
		"vmovdqa ymm0, [%q[shuffle_mask] + r8];"
		"vpermd ymm0, ymm0, ymm1;"
		"vmovdqu [%q[result] + %q[count]*4], ymm0;"

		"popcnt r9d, r9d;"
		"add %q[count], r9;"

 		"jmp 1b;"
#if IACA_INTERSECT_AVX2
		IACA_END_ASM
#endif
	"2: "
		".att_syntax;"
		: [count]"+r"(count), [i_a]"+r"(i_a), [i_b]"+r"(i_b)
		: [st_a]"r"(st_a), [st_b]"r"(st_b),
			[list1]"r"(list1), [list2]"r"(list2),
			[result]"r"(result), [shuffle_mask]"r"(shuffle_mask_avx)
		: "%rax", "%rbx", "%r8", "%r9",
			"ymm0","ymm1","ymm2","ymm3","ymm4",
			"ymm5","ymm6","ymm7","ymm8","ymm9",
			"ymm10","ymm11","ymm12","ymm13","ymm14","ymm15",
			"memory", "cc"
	);
	// intersect the tail using scalar intersection
	count += intersect_scalar_branchless(
		list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count
	);
	return count;
}

size_t intersect_vector_avx2_asm_count(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2){
	size_t count=0, i_a=0, i_b=0;
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;

	asm(".intel_syntax noprefix;"

		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor r9, r9;"
#if IACA_INTERSECT_AVX2_COUNT
		IACA_START_ASM
#endif
	"1: "
		"cmp %[i_a], %[st_a];"
		"je 2f;"
		"cmp %[i_b], %[st_b];"
		"je 2f;"

		"vmovdqa ymm1, [%q[list1] + %q[i_a]*4];" // elements are 4 byte
		"vmovdqa ymm2, [%q[list2] + %q[i_b]*4];"

		"mov r8d, [%q[list1] + %q[i_a]*4 + 28];" //int32_t a_max = list1[i_a+7];
		"cmp r8d, [%q[list2] + %q[i_b]*4 + 28];"
		"setbe al;"
		"setae bl;"
		"lea %q[i_a], [%q[i_a] + rax*8];"
		"lea %q[i_b], [%q[i_b] + rbx*8];"

		"vpcmpeqd ymm10, ymm1, ymm2;"
		"vperm2f128 ymm6, ymm2, ymm2, 1;"
		"vpermilps ymm3, ymm2, 0x39;"
		"vpermilps ymm4, ymm2, 0x4e;"
		"vpcmpeqd ymm11, ymm1, ymm3;"
		"vpermilps ymm5, ymm2, 0x93;"
		"vpcmpeqd ymm4, ymm1, ymm4;"
		"vpermilps ymm7, ymm6, 0x39;"
		"vpermilps ymm8, ymm6, 0x4e;"
		"vpermilps ymm9, ymm6, 0x93;"
		"vpcmpeqd ymm5, ymm1, ymm5;"

		"vpor ymm10, ymm10, ymm11;"
		"vpcmpeqd ymm12, ymm1, ymm6;"
		"vpor ymm4, ymm4, ymm5;"

		"vpcmpeqd ymm13, ymm1, ymm7;"
		"vpcmpeqd ymm8, ymm1, ymm8;"
		"vpcmpeqd ymm9, ymm1, ymm9;"

		"vpor ymm12, ymm12, ymm13;"
		"vpor ymm8, ymm8, ymm9;"

		"vpor ymm10, ymm10, ymm4;"
		"vpor ymm12, ymm12, ymm8;"

		"vpor ymm10, ymm10, ymm12;"

		"vmovmskps r9d, ymm10;"

		"popcnt r9d, r9d;"
		"add %q[count], r9;"

		"jmp 1b;"
#if IACA_INTERSECT_AVX2_COUNT
		IACA_END_ASM
#endif
	"2: "
		".att_syntax;"
		: [count]"+r"(count), [i_a]"+r"(i_a), [i_b]"+r"(i_b)
		: [st_a]"r"(st_a), [st_b]"r"(st_b),
			[list1]"r"(list1), [list2]"r"(list2)
		: "%rax", "%rbx", "%r8", "%r9",
			"ymm0","ymm1","ymm2","ymm3","ymm4",
			"ymm5","ymm6","ymm7","ymm8","ymm9",
			"ymm10","ymm11","ymm12","ymm13","ymm14","ymm15",
			"memory", "cc"
	);
	// intersect the tail using scalar intersection
	count += intersect_scalar_branchless_count(
		list1+i_a, size1-i_a, list2+i_b, size2-i_b
	);
	return count;
}

#endif
