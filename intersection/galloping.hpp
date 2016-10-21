#ifndef INTERSECTION_GALLOPING_HPP_
#define INTERSECTION_GALLOPING_HPP_

#include <immintrin.h>


// a naive version inspired by this paper:
// "SIMD Compression and the Intersection of Sorted Integers"
// source code: https://github.com/lemire/SIMDCompressionAndIntersection
size_t intersect_galloping_V1_AVX(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
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
size_t intersect_galloping_V1_AVX_count(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2){
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

size_t intersect_galloping_V1_SSE(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
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
size_t intersect_galloping_V1_SSE_count(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2){
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


#endif
