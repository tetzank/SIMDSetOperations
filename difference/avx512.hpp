#ifndef DIFFERENCE_AVX512_HPP_
#define DIFFERENCE_AVX512_HPP_

#include <immintrin.h>

#include "naive.hpp"


// set difference with vpconflictd
//
// get 256-bit from lists, combine, first operand in upper part
// vpconflictd -> conflicts marked in upper part as it compares with slots before
// invert mask (done with cmpeq instead of cmpneq as in intersection), only upper part, compressstore upper part with mask
size_t difference_vector_avx512_conflict(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;

	__mmask16 kmask=0xffff, kupper=0xff00;
	__m512i vzero = _mm512_setzero_epi32();
	while(i_a < st_a && i_b < st_b){
		__m256i v_a = _mm256_load_si256((__m256i*)&list1[i_a]);
		__m256i v_b = _mm256_load_si256((__m256i*)&list2[i_b]);

		int32_t a_max = list1[i_a+7];
		int32_t b_max = list2[i_b+7];
		i_a += (a_max <= b_max) * 8;
		i_b += (a_max >= b_max) * 8;

		__mmask16 kwrite = -(uint32_t)(a_max <= b_max); // only write if first list is stepped

		__m512i vpool = _mm512_inserti32x8(_mm512_castsi256_si512(v_b), v_a, 1);
		__m512i vconflict = _mm512_conflict_epi32(vpool);
		__mmask16 kconflict = _mm512_mask_cmpeq_epi32_mask(kupper, vconflict, vzero);
		//TODO: no need to use mask registers everywhere
		kmask = _mm512_kand(kmask, kconflict);  //kmask &= kconflict;
		kconflict = _mm512_kand(kmask, kwrite); //kconflict = kmask & kwrite;
		kmask = _mm512_kor(kmask, kwrite);      //kmask |= kwrite;
		//
		_mm512_mask_compressstoreu_epi32(&result[count], kconflict, vpool);

		count += _mm_popcnt_u32(kconflict);
	}
	uint16_t mask = ~((uint16_t)kmask) >> 8;
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
	count += difference_scalar(
		list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count
	);

#endif
	return count;
}

size_t difference_vector_avx512_conflict_asm(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count=0, i_a=0, i_b=0;
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
	size_t st_a = (size1 / 8) * 8;
	size_t st_b = (size2 / 8) * 8;
	uint16_t mask;

	asm(".intel_syntax noprefix;"

		//"mov eax, 65280;"
		"mov eax, 0xff00;"
		"kmovw k2, eax;"
		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor r9, r9;"
		"vpxord zmm0, zmm0, zmm0;"
		"kxnorw k4, k4, k4;"
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

		"mov r9d, eax;"
		"neg r9d;"
		"kmovw k3, r9d;"

		"vinserti32x8 zmm3, zmm2, ymm1, 1;" // combine to one zmm
		"vpconflictd zmm4, zmm3;" //"vpconflictd zmm4%{k3%}%{z%}, zmm3;"
		"vpcmpeqd k1%{k2%}, zmm4, zmm0;"
		//
		"kandw k4, k4, k1;"
		"kandw k1, k4, k3;" // only store if first list is stepped
		"korw k4, k3, k4;" // reset if we store
		// only write when first list is stepped forward
		"vpcompressd [%q[result] + %q[count]*4] %{k1%}, zmm3;"
		"kmovw r9d, k1;"

		"popcnt r9d, r9d;"
		"add %q[count], r9;"

 		"jmp 1b;"
	"2: "
		// save k4 and handle in tail
		"kmovw %k[mask], k4;"

		".att_syntax;"
		: [count]"+r"(count), [i_a]"+r"(i_a), [i_b]"+r"(i_b), [mask]"=r"(mask)
		: [st_a]"r"(st_a), [st_b]"r"(st_b),
			[list1]"r"(list1), [list2]"r"(list2),
			[result]"r"(result)
		: "%rax", "%rbx", "%r8", "%r9",
			"zmm0","zmm1","zmm2","zmm3","zmm4",
			"memory", "cc"
	);
	mask = ~mask >> 8;
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
	count += difference_scalar(
		list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count
	);
#endif
	return count;
}

#endif
