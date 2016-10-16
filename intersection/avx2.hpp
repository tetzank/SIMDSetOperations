#ifndef INTERSECTION_AVX2_HPP_
#define INTERSECTION_AVX2_HPP_

#include <immintrin.h>


size_t intersect_vector_avx2(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
#ifdef __AVX2__
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
		//__m256 vfa = _mm256_cvtepi32_ps(v_a);
		//__m256 vfb = _mm256_cvtepi32_ps(v_b);

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
		// just use unchanged v_a, don't convert back vfa
		// AVX2: _mm256_maskstore_epi32 directly with cmp_mask
		// just use float variant
// 		_mm256_maskstore_ps((float*)&result[count], (__m256i)cmp_mask, (__m256)v_a);
// 		_mm256_store_si256();
		__m256i idx = _mm256_load_si256((const __m256i*)&shuffle_mask_avx[mask*8]);
		//__m256i p = (__m256i)permute8x32((__m256)v_a, idx);
		__m256i p = _mm256_permutevar8x32_epi32(v_a, idx);
		_mm256_storeu_si256((__m256i*)&result[count], p);

		count += _mm_popcnt_u32(mask);
	}
	// intersect the tail using scalar intersection
	count += intersect_scalar(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

#endif
	return count;
}
size_t intersect_vector_avx2_count(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2){
	size_t count=0, i_a=0, i_b=0;
#ifdef __AVX2__
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

#endif
	return count;
}


size_t intersect_vector_avx2_asm(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2, uint32_t *result){
	//TODO
	return 0;
}
size_t intersect_vector_avx2_asm_count(uint32_t *list1, size_t size1, uint32_t *list2, size_t size2){
	size_t count=0, i_a=0, i_b=0;
#ifdef __AVX2__
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

		"vmovdqa ymm1, [%q6 + %q1*4];" // elements are 4 byte
		"vmovdqa ymm2, [%q7 + %q2*4];"

		"mov r8d, [%q6 + %q1*4 + 28];" //int32_t a_max = list1[i_a+7];
		"cmp r8d, [%q7 + %q2*4 + 28];"
		"setle al;"
		"setge bl;"
		"lea %q1, [%q1 + rax*8];"
		"lea %q2, [%q2 + rbx*8];"

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

#endif
