#ifndef INTERSECTION_AVX512_HPP_
#define INTERSECTION_AVX512_HPP_

#include <immintrin.h>

#include "naive.hpp"
#include "branchless.hpp"

#include "../shuffle_dictionary.hpp"


// simple intersection with vpconflictd
//
// load 256-bit from each list, check for common elements with vpconflictd
// convert result vector to mask, compressstore
size_t intersect_vector_avx512_conflict(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;

	__m512i vzero = _mm512_setzero_epi32();
	while(i_a < st_a && i_b < st_b){
		__m256i v_a = _mm256_load_si256((__m256i*)&list1[i_a]);
		__m256i v_b = _mm256_load_si256((__m256i*)&list2[i_b]);

		int32_t a_max = list1[i_a+7];
		int32_t b_max = list2[i_b+7];
		i_a += (a_max <= b_max) * 8;
		i_b += (a_max >= b_max) * 8;
	
		__m512i vpool = _mm512_inserti32x8(_mm512_castsi256_si512(v_a), v_b, 1);
		__m512i vconflict = _mm512_conflict_epi32(vpool);
		// _mm512_movepi32_mask doesn't work, use comparison with zero
		__mmask16 kconflict = _mm512_cmpneq_epi32_mask(vconflict, vzero);

		_mm512_mask_compressstoreu_epi32(&result[count], kconflict, vpool);

		count += _mm_popcnt_u32(kconflict);
	}
	// intersect the tail using scalar intersection
	count += intersect_scalar(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

#endif
	return count;
}

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


static constexpr constarray<uint32_t,16*15> prepare_shuffle_vectors(){
	constarray<uint32_t,16*15> arr = {};
	uint32_t start=1;
	for(uint32_t i=0; i<15; ++i){
		uint32_t counter = start;
		for(uint32_t j=0; j<16; ++j){
			arr[i*16 + j] = counter % 16;
			++counter;
		}
		++start;
	}
	return arr;
}
static const constexpr auto shuffle_vectors_arr = prepare_shuffle_vectors();
static const constexpr __m512i *shuffle_vectors = (__m512i*)shuffle_vectors_arr.elems;

size_t intersect_vector_avx512(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
	size_t st_a = (size1 / 16) * 16;
	size_t st_b = (size2 / 16) * 16;

	__m512i sv0  = shuffle_vectors[ 0];//_mm512_set_epi32(0,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);
	__m512i sv1  = shuffle_vectors[ 1];//_mm512_set_epi32(1,0,15,14,13,12,11,10,9,8,7,6,5,4,3,2);
	__m512i sv2  = shuffle_vectors[ 2];//_mm512_set_epi32(2,1,0,15,14,13,12,11,10,9,8,7,6,5,4,3);
	__m512i sv3  = shuffle_vectors[ 3];//_mm512_set_epi32(3,2,1,0,15,14,13,12,11,10,9,8,7,6,5,4);
	__m512i sv4  = shuffle_vectors[ 4];//_mm512_set_epi32(4,3,2,1,0,15,14,13,12,11,10,9,8,7,6,5);
	__m512i sv5  = shuffle_vectors[ 5];//_mm512_set_epi32(5,4,3,2,1,0,15,14,13,12,11,10,9,8,7,6);
	__m512i sv6  = shuffle_vectors[ 6];//_mm512_set_epi32(6,5,4,3,2,1,0,15,14,13,12,11,10,9,8,7);
	__m512i sv7  = shuffle_vectors[ 7];//_mm512_set_epi32(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8);
	__m512i sv8  = shuffle_vectors[ 8];//_mm512_set_epi32(8,7,6,5,4,3,2,1,0,15,14,13,12,11,10,9);
	__m512i sv9  = shuffle_vectors[ 9];//_mm512_set_epi32(9,8,7,6,5,4,3,2,1,0,15,14,13,12,11,10);
	__m512i sv10 = shuffle_vectors[10];//_mm512_set_epi32(10,9,8,7,6,5,4,3,2,1,0,15,14,13,12,11);
	__m512i sv11 = shuffle_vectors[11];//_mm512_set_epi32(11,10,9,8,7,6,5,4,3,2,1,0,15,14,13,12);
	__m512i sv12 = shuffle_vectors[12];//_mm512_set_epi32(12,11,10,9,8,7,6,5,4,3,2,1,0,15,14,13);
	__m512i sv13 = shuffle_vectors[13];//_mm512_set_epi32(13,12,11,10,9,8,7,6,5,4,3,2,1,0,15,14);
	__m512i sv14 = shuffle_vectors[14];//_mm512_set_epi32(14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,15);

	//__m512i vzero = _mm512_setzero_epi32();
	while(i_a < st_a && i_b < st_b){
		__m512i v_a = _mm512_load_epi32((__m512i*)&list1[i_a]);
		__m512i v_b = _mm512_load_epi32((__m512i*)&list2[i_b]);

		int32_t a_max = list1[i_a+15];
		int32_t b_max = list2[i_b+15];
		i_a += (a_max <= b_max) * 16;
		i_b += (a_max >= b_max) * 16;

		__mmask16 cmp0 = _mm512_cmpeq_epi32_mask(v_a, v_b);
		__m512i rot0 = _mm512_permutexvar_epi32(sv0, v_b);
		__mmask16 cmp1 = _mm512_cmpeq_epi32_mask(v_a, rot0);
		__m512i rot1 = _mm512_permutexvar_epi32(sv1, v_b);
		__mmask16 cmp2 = _mm512_cmpeq_epi32_mask(v_a, rot1);
		__m512i rot2 = _mm512_permutexvar_epi32(sv2, v_b);
		__mmask16 cmp3 = _mm512_cmpeq_epi32_mask(v_a, rot2);
		cmp0 = _mm512_kor(_mm512_kor(cmp0, cmp1), _mm512_kor(cmp2, cmp3));

		__m512i rot3 = _mm512_permutexvar_epi32(sv3, v_b);
		__mmask16 cmp4 = _mm512_cmpeq_epi32_mask(v_a, rot3);
		__m512i rot4 = _mm512_permutexvar_epi32(sv4, v_b);
		__mmask16 cmp5 = _mm512_cmpeq_epi32_mask(v_a, rot4);
		__m512i rot5 = _mm512_permutexvar_epi32(sv5, v_b);
		__mmask16 cmp6 = _mm512_cmpeq_epi32_mask(v_a, rot5);
		__m512i rot6 = _mm512_permutexvar_epi32(sv6, v_b);
		__mmask16 cmp7 = _mm512_cmpeq_epi32_mask(v_a, rot6);
		cmp4 = _mm512_kor(_mm512_kor(cmp4, cmp5), _mm512_kor(cmp6, cmp7));

		__m512i rot7 = _mm512_permutexvar_epi32(sv7, v_b);
		/*__mmask16 cmp8*/cmp1 = _mm512_cmpeq_epi32_mask(v_a, rot7);
		__m512i rot8 = _mm512_permutexvar_epi32(sv8, v_b);
		/*__mmask16 cmp9*/cmp2 = _mm512_cmpeq_epi32_mask(v_a, rot8);
		__m512i rot9 = _mm512_permutexvar_epi32(sv9, v_b);
		/*__mmask16 cmp10*/cmp3 = _mm512_cmpeq_epi32_mask(v_a, rot9);
		__m512i rot10 = _mm512_permutexvar_epi32(sv10, v_b);
		/*__mmask16 cmp11*/cmp5 = _mm512_cmpeq_epi32_mask(v_a, rot10);
		cmp1 = _mm512_kor(_mm512_kor(cmp1, cmp2), _mm512_kor(cmp3, cmp5));

		__m512i rot11 = _mm512_permutexvar_epi32(sv11, v_b);
		/*__mmask16 cmp12*/cmp2 = _mm512_cmpeq_epi32_mask(v_a, rot11);
		__m512i rot12 = _mm512_permutexvar_epi32(sv12, v_b);
		/*__mmask16 cmp13*/cmp3 = _mm512_cmpeq_epi32_mask(v_a, rot12);
		__m512i rot13 = _mm512_permutexvar_epi32(sv13, v_b);
		/*__mmask16 cmp14*/cmp5 = _mm512_cmpeq_epi32_mask(v_a, rot13);
		__m512i rot14 = _mm512_permutexvar_epi32(sv14, v_b);
		/*__mmask16 cmp15*/cmp6 = _mm512_cmpeq_epi32_mask(v_a, rot14);
		cmp2 = _mm512_kor(_mm512_kor(cmp2, cmp3), _mm512_kor(cmp5, cmp6));

		cmp0 = _mm512_kor(_mm512_kor(cmp0, cmp4), _mm512_kor(cmp1, cmp2));
		//__mmask16 cmp =
		//	(
		//		((cmp0  | cmp1 ) | (cmp2  | cmp3 ))
		//    	|
		//		((cmp4  | cmp5 ) | (cmp6  | cmp7 ))
		//	) | (
		//		((cmp8  | cmp9 ) | (cmp10 | cmp11))
		//		|
		//		((cmp12 | cmp13) | (cmp14 | cmp15))
		//	);
		//__mmask16 cmp = _mm512_kor(
		//	_mm512_kor(
		//		_mm512_kor(_mm512_kor(cmp0, cmp1), _mm512_kor(cmp2, cmp3 ))
		//    	,
		//		_mm512_kor(_mm512_kor(cmp4, cmp5), _mm512_kor(cmp6, cmp7))
		//	), _mm512_kor(
		//		_mm512_kor(_mm512_kor(cmp8, cmp9 ), _mm512_kor(cmp10, cmp11))
		//		,
		//		_mm512_kor(_mm512_kor(cmp12, cmp13), _mm512_kor(cmp14, cmp15))
		//	)
		//);
		_mm512_mask_compressstoreu_epi32(&result[count], cmp0, v_a);
		count += _mm_popcnt_u32(cmp0);
	}
	// intersect the tail using scalar intersection
	count += intersect_scalar(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

#endif
	return count;
}

size_t intersect_vector_avx512_asm(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
	size_t st_a = (size1 / 16) * 16;
	size_t st_b = (size2 / 16) * 16;


	asm(".intel_syntax noprefix;"

		"vmovdqa32 zmm0 , [%[shuffle_vectors]];"
		"vmovdqa32 zmm1 , [%[shuffle_vectors] + 0x040];"
		"vmovdqa32 zmm2 , [%[shuffle_vectors] + 0x080];"
		"vmovdqa32 zmm3 , [%[shuffle_vectors] + 0x0c0];"
		"vmovdqa32 zmm4 , [%[shuffle_vectors] + 0x100];"
		"vmovdqa32 zmm5 , [%[shuffle_vectors] + 0x140];"
		"vmovdqa32 zmm6 , [%[shuffle_vectors] + 0x180];"
		"vmovdqa32 zmm7 , [%[shuffle_vectors] + 0x1c0];"
		"vmovdqa32 zmm8 , [%[shuffle_vectors] + 0x200];"
		"vmovdqa32 zmm9 , [%[shuffle_vectors] + 0x240];"
		"vmovdqa32 zmm10, [%[shuffle_vectors] + 0x280];"
		"vmovdqa32 zmm11, [%[shuffle_vectors] + 0x2c0];"
		"vmovdqa32 zmm12, [%[shuffle_vectors] + 0x300];"
		"vmovdqa32 zmm13, [%[shuffle_vectors] + 0x340];"
		"vmovdqa32 zmm14, [%[shuffle_vectors] + 0x380];"

		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor r9, r9;"
	"1: "
		"cmp %[i_a], %[st_a];"
		"je 2f;"
		"cmp %[i_b], %[st_b];"
		"je 2f;"

		"vmovdqa32 zmm15, [%q[list1] + %q[i_a]*4];" // elements are 4 byte
		"vmovdqa32 zmm16, [%q[list2] + %q[i_b]*4];"

		// increase i_a and i_b
		"mov r8d, [%q[list1] + %q[i_a]*4 + 60];" // int32_t a_max = list1[i_a+15];
		"cmp r8d, [%q[list2] + %q[i_b]*4 + 60];" // 15*4 = 60
		"setle al;"
		"setge bl;"
		//"lea %q[i_a], [%q[i_a] + rax*8];" //no *16 in address mode
		//"lea %q[i_b], [%q[i_b] + rbx*8];"
		"shl rax, 4;"
		"shl rbx, 4;"
		"add %q[i_a], rax;"
		"add %q[i_b], rbx;"

		"vpcmpeqd k1, zmm15, zmm16;"
		"vpermd zmm17, zmm0, zmm16;"
		"vpcmpeqd k2, zmm15, zmm17;"
		"vpermd zmm18, zmm1, zmm16;"
		"vpcmpeqd k3, zmm15, zmm18;"
		"vpermd zmm19, zmm2, zmm16;"
		"vpcmpeqd k4, zmm15, zmm19;"

		"korw k1, k1, k2;"
		"korw k3, k3, k4;"
		"korw k1, k1, k3;"

		"vpermd zmm20, zmm3, zmm16;"
		"vpcmpeqd k5, zmm15, zmm20;"
		"vpermd zmm21, zmm4, zmm16;"
		"vpcmpeqd k6, zmm15, zmm21;"
		"vpermd zmm22, zmm5, zmm16;"
		"vpcmpeqd k7, zmm15, zmm22;"
		"vpermd zmm23, zmm6, zmm16;"
		"vpcmpeqd k2, zmm15, zmm23;"

		"korw k5, k5, k6;"
		"korw k7, k7, k2;"
		"korw k5, k5, k7;"

		"vpermd zmm24, zmm7, zmm16;"
		"vpcmpeqd k3, zmm15, zmm24;"
		"vpermd zmm25, zmm8, zmm16;"
		"vpcmpeqd k4, zmm15, zmm25;"
		"vpermd zmm26, zmm9, zmm16;"
		"vpcmpeqd k6, zmm15, zmm26;"
		"vpermd zmm27, zmm10, zmm16;"
		"vpcmpeqd k7, zmm15, zmm27;"

		"korw k3, k3, k4;"
		"korw k6, k6, k7;"
		"korw k3, k3, k6;"

		"vpermd zmm28, zmm11, zmm16;"
		"vpcmpeqd k2, zmm15, zmm28;"
		"vpermd zmm29, zmm12, zmm16;"
		"vpcmpeqd k4, zmm15, zmm29;"
		"vpermd zmm30, zmm13, zmm16;"
		"vpcmpeqd k6, zmm15, zmm30;"
		"vpermd zmm31, zmm14, zmm16;"
		"vpcmpeqd k7, zmm15, zmm31;"

		"korw k2, k2, k4;"
		"korw k6, k6, k7;"
		"korw k2, k2, k6;"

		"korw k1, k1, k5;"
		"korw k3, k3, k2;"
		"korw k1, k1, k3;"

		"vpcompressd [%q[result] + %q[count]*4] %{k1%}, zmm15;"
		"kmovw r9d, k1;"

		"popcnt r9d, r9d;"
		"add %q[count], r9;"

		"jmp 1b;"
	"2: "
		".att_syntax;"
		: [count]"+r"(count), [i_a]"+r"(i_a), [i_b]"+r"(i_b)
		: [st_a]"r"(st_a), [st_b]"r"(st_b),
			[list1]"r"(list1), [list2]"r"(list2),
			[result]"r"(result), [shuffle_vectors]"r"(shuffle_vectors)
		: "%rax", "%rbx", "%r8", "%r9",
			"zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7",
			"zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",
			"zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23",
			"zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31",
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
