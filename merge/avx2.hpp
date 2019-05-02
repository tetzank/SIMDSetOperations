#ifndef MERGE_AVX2_HPP_
#define MERGE_AVX2_HPP_

#include <immintrin.h>


size_t merge_vector_avx2_bitonic(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count = 0;
	size_t i_a = 0, i_b = 0;
	// trim lengths to be a multiple of 16
	size_t st_a = ((size1-1) / 8) * 8;
	size_t st_b = ((size2-1) / 8) * 8;
	//
	uint32_t a_nextfirst, b_nextfirst;
	alignas(32) uint32_t maxtail[8];

	if(i_a < st_a && i_b < st_b){
		__m256i vreverse = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
		__m256i vminshuf = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
		__m256i vmaxshuf = _mm256_set_epi32(0, 4, 1, 5, 2, 6, 3, 7); // reverse as well

		__m256i v_a = _mm256_load_si256((__m256i*)list1);
		__m256i vb  = _mm256_load_si256((__m256i*)list2);
		__m256i v_b = _mm256_permutevar8x32_epi32(vb, vreverse);
		do{
			// bitonic merge network
			// level 1
			__m256i min = _mm256_min_epi32(v_a, v_b);
			__m256i max = _mm256_max_epi32(v_a, v_b);
			__m256i L = _mm256_blend_epi32(min, max, 0b11110000);
			__m256i H = _mm256_permute2x128_si256(min, max, 0x21);
			// level 2
			min = _mm256_min_epi32(L, H);
			max = _mm256_max_epi32(L, H);
			L = _mm256_blend_epi32(min, max, 0b11001100);
			H = (__m256i)_mm256_shuffle_ps((__m256)min, (__m256)max, _MM_SHUFFLE(1,0,3,2));
			// level 3
			min = _mm256_min_epi32(L, H);
			max = _mm256_max_epi32(L, H);
			L = _mm256_blend_epi32(min, max, 0b10101010);
			H = _mm256_blend_epi32(min, max, 0b01010101);
			H = _mm256_shuffle_epi32(H, _MM_SHUFFLE(2,3,0,1));
			// level 4
			min = _mm256_min_epi32(L, H);
			max = _mm256_max_epi32(L, H);
			L = _mm256_permute2x128_si256(min, max, 0x20);
			L = _mm256_permutevar8x32_epi32(L, vminshuf);
			H = _mm256_permute2x128_si256(min, max, 0x31);
			H = _mm256_permutevar8x32_epi32(H, vmaxshuf); // reverses for next iteration

			_mm256_store_si256((__m256i*)&result[count], L);
			count += 8;

			v_a = H;
			// compare first element of the next block in both lists
			a_nextfirst = list1[i_a+8];
			b_nextfirst = list2[i_b+8];
			// write minimum as above out to result
			// keep maximum and do the same steps as above with next block
			// next block from one list, which first element in new block is smaller
			i_a += (a_nextfirst <= b_nextfirst) * 8;
			i_b += (a_nextfirst > b_nextfirst) * 8;
			size_t index = (a_nextfirst <= b_nextfirst)? i_a: i_b;
			const uint32_t *base = (a_nextfirst <= b_nextfirst)? list1: list2;
			v_b = _mm256_load_si256((__m256i*)&base[index]);
		}while(i_a < st_a && i_b < st_b);

		// v_a contains max vector from last comparison, v_b contains new, might be out of bounds
		// indices i_a and i_b correct, still need to handle v_a
		_mm256_store_si256((__m256i*)maxtail, _mm256_permutevar8x32_epi32(v_a, vreverse));

		size_t mti=0;
		size_t mtsize = 8;
		if(a_nextfirst <= b_nextfirst){
			// compare maxtail with list1
			while(mti < mtsize && i_a < size1){
				if(maxtail[mti] < list1[i_a]){
					result[count++] = maxtail[mti];
					mti++;
				}else{
					result[count++] = list1[i_a];
					i_a++;
				}
			}
			i_b += 8;
		}else{
			// compare maxtail with list2
			while(mti < mtsize && i_b < size2){
				if(maxtail[mti] < list2[i_b]){
					result[count++] = maxtail[mti];
					mti++;
				}else{
					result[count++] = list2[i_b];
					i_b++;
				}
			}
			i_a += 8;
		}
		while(mti < mtsize){
			result[count++] = maxtail[mti++];
		}
	}
	// scalar tail
	count += merge_scalar(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);
	return count;
}

#endif
