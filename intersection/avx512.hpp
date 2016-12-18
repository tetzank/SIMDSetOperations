#ifndef INTERSECTION_AVX512_HPP_
#define INTERSECTION_AVX512_HPP_

#include <immintrin.h>

#include "branchless.hpp"


size_t intersect_vector_avx512_conflict_asm(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;

	asm(".intel_syntax noprefix;"

		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor r9, r9;"
		"vpxord zmm0, zmm0, zmm0;"
	"1: "
 		"cmp %[i_a], %[st_a];"
 		"je 2f;"
		"cmp %[i_b], %[st_b];"
		"je 2f;"

		"vmovdqa ymm1, [%q[list1] + %q[i_a]*4];" // elements are 4 byte
		"vmovdqa ymm2, [%q[list2] + %q[i_b]*4];"

		"mov r8d, [%q[list1] + %q[i_a]*4 + 28];" //int32_t a_max = list1[i_a+7];
		"cmp r8d, [%q[list2] + %q[i_b]*4 + 28];"
		"setle al;"
		"setge bl;"
		"lea %q[i_a], [%q[i_a] + rax*8];"
		"lea %q[i_b], [%q[i_b] + rbx*8];"

		//FIXME: vinserti32x8 only available in avx512dq which KNL doesn't have
		"vinserti32x8 zmm3, zmm1, ymm2, 1;" // combine to one zmm
		"vpconflictd zmm4, zmm3;"
		"vpcmpneqd k1, zmm4, zmm0;"
		"vpcompressd [%q[result] + %q[count]*4] %{k1%}, zmm3;"
		"kmovw r9d, k1;"

		"popcnt r9d, r9d;"
		"add %q[count], r9;"

 		"jmp 1b;"
	"2: "
		".att_syntax;"
		: [count]"+r"(count), [i_a]"+r"(i_a), [i_b]"+r"(i_b)
		: [st_a]"r"(st_a), [st_b]"r"(st_b),
			[list1]"r"(list1), [list2]"r"(list2),
			[result]"r"(result)//, [shuffle_mask]"r"(shuffle_mask_avx)
		: "%rax", "%rbx", "%r8", "%r9",
			"zmm0","zmm1","zmm2","zmm3","zmm4",
			/*"ymm5","ymm6","ymm7","ymm8","ymm9",
			"ymm10","ymm11","ymm12","ymm13","ymm14","ymm15",*/
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
