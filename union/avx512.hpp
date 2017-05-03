#ifndef UNION_AVX512_HPP_
#define UNION_AVX512_HPP_

#include <immintrin.h>

#include <algorithm>

#include "branchless.hpp"


// union based on bitonic merge, always using the 2-vector shuffle for now
#define L 0
#define H 16
alignas(64) static uint32_t shuffles[][16]={
	{ 0|L, 1|L, 2|L, 3|L, 4|L, 5|L, 6|L, 7|L,  0|H, 1|H, 2|H, 3|H, 4|H, 5|H, 6|H, 7|H }, // L1->L2 for L
	{ 8|L, 9|L,10|L,11|L,12|L,13|L,14|L,15|L,  8|H, 9|H,10|H,11|H,12|H,13|H,14|H,15|H }, // L1->L2 for H

	{ 0|L, 1|L, 2|L, 3|L, 0|H, 1|H, 2|H, 3|H,  8|L, 9|L,10|L,11|L, 8|H, 9|H,10|H,11|H }, // L2->L3 for L
	{ 4|L, 5|L, 6|L, 7|L, 4|H, 5|H, 6|H, 7|H, 12|L,13|L,14|L,15|L,12|H,13|H,14|H,15|H }, // L2->L3 for H

	{ 0|L, 1|L, 0|H, 1|H, 4|L, 5|L, 4|H, 5|H,  8|L, 9|L, 8|H, 9|H,12|L,13|L,12|H,13|H }, // L3->L4 for L
	{ 2|L, 3|L, 2|H, 3|H, 6|L, 7|L, 6|H, 7|H, 10|L,11|L,10|H,11|H,14|L,15|L,14|H,15|H }, // L3->L4 for H

	{ 0|L, 0|H, 2|L, 2|H, 4|L, 4|H, 6|L, 6|H,  8|L, 8|H,10|L,10|H,12|L,12|H,14|L,14|H }, // L4->L5 for L
	{ 1|L, 1|H, 3|L, 3|H, 5|L, 5|H, 7|L, 7|H,  9|L, 9|H,11|L,11|H,13|L,13|H,15|L,15|H }, // L4->L5 for H

	{ 0|L, 0|H, 1|L, 1|H, 2|L, 2|H, 3|L, 3|H,  4|L, 4|H, 5|L, 5|H, 6|L, 6|H, 7|L, 7|H }, // output first 16 elements
	//{ 8|L, 8|H, 9|L, 9|H,10|L,10|H,11|L,11|H, 12|L,12|H,13|L,13|H,14|L,14|H,15|L,15|H }, // output second 16 elements
	{15|H,15|L,14|H,14|L,13|H,13|L,12|H,12|L, 11|H,11|L,10|H,10|L, 9|H, 9|L, 8|H, 8|L }
	// output for second reversed for next iteration
};
#undef L
#undef H

alignas(64) static uint32_t reverseshuffle[]={ 15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };

// one shuffle can be replaced with a simple blend, as min/max are commutative
static int blendmasks[] = {
	0b11111111'00000000, // L1->L2
	0b11110000'11110000, // L2->L3
	0b11001100'11001100, // L3->L4
	0b10101010'10101010, // L4->L5
};
// changed shuffles for blend+shuflle per level
#define L 0
#define H 16
alignas(64) static uint32_t shuffles2[][16]={
	{ 8|L, 9|L,10|L,11|L,12|L,13|L,14|L,15|L,  0|H, 1|H, 2|H, 3|H, 4|H, 5|H, 6|H, 7|H }, // L1->L2 for H
	{ 4|L, 5|L, 6|L, 7|L, 0|H, 1|H, 2|H, 3|H, 12|L,13|L,14|L,15|L, 8|H, 9|H,10|H,11|H }, // L2->L3 for H
	{ 2|L, 3|L, 0|H, 1|H, 6|L, 7|L, 4|H, 5|H, 10|L,11|L, 8|H, 9|H,14|L,15|L,12|H,13|H }, // L3->L4 for H
	{ 1|L, 0|H, 3|L, 2|H, 5|L, 4|H, 7|L, 6|H,  9|L, 8|H,11|L,10|H,13|L,12|H,15|L,14|H }, // L4->L5 for H

	{ 0|L, 0|H, 1|L, 1|H, 2|L, 2|H, 3|L, 3|H,  4|L, 4|H, 5|L, 5|H, 6|L, 6|H, 7|L, 7|H }, // output first 16 elements
	{15|H,15|L,14|H,14|L,13|H,13|L,12|H,12|L, 11|H,11|L,10|H,10|L, 9|H, 9|L, 8|H, 8|L }
	// output for second reversed for next iteration
};
#undef L
#undef H


#define BLEND 1

size_t union_vector_avx512_bitonic(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count = 0;
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
	size_t i_a = 0, i_b = 0;
	// trim lengths to be a multiple of 16
	size_t st_a = ((size1-1) / 16) * 16;
	size_t st_b = ((size2-1) / 16) * 16;
	// assumes ~0 -> all bits set is not used
	uint32_t a_nextfirst, b_nextfirst;
#if !BLEND
	uint32_t endofblock=~0;
#else
	__m512i old = _mm512_set1_epi32(-1); //FIXME: hardcoded, use something related to the lists
#endif
	alignas(64) uint32_t maxtail[16];

	if(i_a < st_a && i_b < st_b){
		// load all the shuffles
		__m512i vL1L2_L = _mm512_load_epi32(shuffles[0]);
		__m512i vL1L2_H = _mm512_load_epi32(shuffles[1]);
		__m512i vL2L3_L = _mm512_load_epi32(shuffles[2]);
		__m512i vL2L3_H = _mm512_load_epi32(shuffles[3]);
		__m512i vL3L4_L = _mm512_load_epi32(shuffles[4]);
		__m512i vL3L4_H = _mm512_load_epi32(shuffles[5]);
		__m512i vL4L5_L = _mm512_load_epi32(shuffles[6]);
		__m512i vL4L5_H = _mm512_load_epi32(shuffles[7]);
		__m512i vL5Out_L = _mm512_load_epi32(shuffles[8]);
		__m512i vL5Out_H = _mm512_load_epi32(shuffles[9]);
		__m512i vreverse = _mm512_load_epi32(reverseshuffle);

		__m512i v_a = _mm512_load_epi32(list1);
		__m512i vb = _mm512_load_epi32(list2);
		__m512i v_b = _mm512_permutexvar_epi32(vreverse, vb);

		do{
			// bitonic merge network
			// level 1
			__m512i min = _mm512_min_epi32(v_a, v_b);
			__m512i max = _mm512_max_epi32(v_a, v_b);
			__m512i L = _mm512_permutex2var_epi32(min, vL1L2_L, max);
			__m512i H = _mm512_permutex2var_epi32(min, vL1L2_H, max);
			// level 2
			min = _mm512_min_epi32(L, H);
			max = _mm512_max_epi32(L, H);
			L = _mm512_permutex2var_epi32(min, vL2L3_L, max);
			H = _mm512_permutex2var_epi32(min, vL2L3_H, max);
			// level 3
			min = _mm512_min_epi32(L, H);
			max = _mm512_max_epi32(L, H);
			L = _mm512_permutex2var_epi32(min, vL3L4_L, max);
			H = _mm512_permutex2var_epi32(min, vL3L4_H, max);
			// level 4
			min = _mm512_min_epi32(L, H);
			max = _mm512_max_epi32(L, H);
			L = _mm512_permutex2var_epi32(min, vL4L5_L, max);
			H = _mm512_permutex2var_epi32(min, vL4L5_H, max);
			// level 5
			min = _mm512_min_epi32(L, H);
			max = _mm512_max_epi32(L, H);
			L = _mm512_permutex2var_epi32(min, vL5Out_L, max);
			H = _mm512_permutex2var_epi32(min, vL5Out_H, max);

#if !BLEND
			// deduplicate over block end, 1 2 3 ... 15 | 15 16 ...
			// get lowest and highest value from vector
			// no extract instruction to get scalar -> compressstore
			uint32_t tmp[16]; // only uses first 2 elements
			_mm512_mask_compressstoreu_epi32(tmp, 0x8001, L);
			uint32_t first = tmp[0];
			count -= (endofblock==first);
			endofblock = tmp[1];
			// deduplicate first 16 elements and compressstore them
			__m512i vconflict = _mm512_conflict_epi32(L);
			__mmask16 kconflict = _mm512_cmpeq_epi32_mask(vconflict, _mm512_setzero_epi32());
			_mm512_mask_compressstoreu_epi32(&result[count], kconflict, L);
			count += _mm_popcnt_u32(kconflict); // number of elements written
#else
			__m512i recon = _mm512_mask_blend_epi32(0x7fff, old, L);
			const __m512i circlic_shuffle = _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15);
			__m512i dedup = _mm512_permutexvar_epi32(circlic_shuffle, recon);
			__mmask16 kmask = _mm512_cmpneq_epi32_mask(dedup, L);
			_mm512_mask_compressstoreu_epi32(&result[count], kmask, L);
			count += _mm_popcnt_u32(kmask); // number of elements written
			// remember minimum for next iteration
			old = L;
#endif

			v_a = H;
			// compare first element of the next block in both lists
			a_nextfirst = list1[i_a+16];
			b_nextfirst = list2[i_b+16];
			// write minimum as above out to result
			// keep maximum and do the same steps as above with next block
			// next block from one list, which first element in new block is smaller
			i_a += (a_nextfirst <= b_nextfirst) * 16;
			i_b += (a_nextfirst > b_nextfirst) * 16;
			size_t index = (a_nextfirst <= b_nextfirst)? i_a: i_b;
			const uint32_t *base = (a_nextfirst <= b_nextfirst)? list1: list2;
			v_b = _mm512_load_epi32(&base[index]);
		}while(i_a < st_a && i_b < st_b);
		
		// v_a contains max vector from last comparison, v_b contains new, might be out of bounds
		// indices i_a and i_b correct, still need to handle v_a
		_mm512_store_epi32(maxtail, _mm512_permutexvar_epi32(vreverse, v_a));
#if BLEND
		uint32_t endofblock = _mm_extract_epi32(_mm512_extracti32x4_epi32(old, 3), 3);
#endif

		size_t mti=0;
		size_t mtsize = std::unique(maxtail, maxtail+16) - maxtail; // deduplicate tail
		if(a_nextfirst <= b_nextfirst){
			// endofblock needs to be considered too, for deduplication
			if(endofblock == std::min(maxtail[0],list1[i_a])) --count;
			// compare maxtail with list1
			while(mti < mtsize && i_a < size1){
				if(maxtail[mti] < list1[i_a]){
					result[count++] = maxtail[mti];
					mti++;
				}else if(maxtail[mti] > list1[i_a]){
					result[count++] = list1[i_a];
					i_a++;
				}else{
					result[count++] = maxtail[mti];
					mti++; i_a++;
				}
			}
			i_b += 16;
		}else{
			// endofblock needs to be considered too, for deduplication
			if(endofblock == std::min(maxtail[0],list2[i_b])) --count;
			// compare maxtail with list2
			while(mti < mtsize && i_b < size2){
				if(maxtail[mti] < list2[i_b]){
					result[count++] = maxtail[mti];
					mti++;
				}else if(maxtail[mti] > list2[i_b]){
					result[count++] = list2[i_b];
					i_b++;
				}else{
					result[count++] = maxtail[mti];
					mti++; i_b++;
				}
			}
			i_a += 16;
		}
		while(mti < mtsize){
			result[count++] = maxtail[mti++];
		}
	}

	// scalar tail
	count += union_scalar_branchless(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

#endif
	return count;
}

// nearly the same as above, only changed to replace one shuffle with a blend
size_t union_vector_avx512_bitonic2(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t count = 0;
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
	size_t i_a = 0, i_b = 0;
	// trim lengths to be a multiple of 16
	size_t st_a = ((size1-1) / 16) * 16;
	size_t st_b = ((size2-1) / 16) * 16;
	// assumes ~0 -> all bits set is not used
	uint32_t a_nextfirst, b_nextfirst;
#if !BLEND
	uint32_t endofblock=~0;
#else
	__m512i old = _mm512_set1_epi32(-1); //FIXME: hardcoded, use something related to the lists
#endif
	alignas(64) uint32_t maxtail[16];

	if(i_a < st_a && i_b < st_b){
		// load all the shuffles
		__m512i vL1L2 = _mm512_load_epi32(shuffles2[0]);
		__m512i vL2L3 = _mm512_load_epi32(shuffles2[1]);
		__m512i vL3L4 = _mm512_load_epi32(shuffles2[2]);
		__m512i vL4L5 = _mm512_load_epi32(shuffles2[3]);
		__m512i vL5Out_L = _mm512_load_epi32(shuffles2[4]);
		__m512i vL5Out_H = _mm512_load_epi32(shuffles2[5]);

		__mmask16 kL1L2 = blendmasks[0];
		__mmask16 kL2L3 = blendmasks[1];
		__mmask16 kL3L4 = blendmasks[2];
		__mmask16 kL4L5 = blendmasks[3];

		__m512i vreverse = _mm512_load_epi32(reverseshuffle);

		__m512i v_a = _mm512_load_epi32(list1);
		__m512i vb = _mm512_load_epi32(list2);
		__m512i v_b = _mm512_permutexvar_epi32(vreverse, vb);

		do{
			// bitonic merge network
			// level 1
			__m512i min = _mm512_min_epi32(v_a, v_b);
			__m512i max = _mm512_max_epi32(v_a, v_b);
			__m512i L = _mm512_mask_blend_epi32(kL1L2, min, max);
			__m512i H = _mm512_permutex2var_epi32(min, vL1L2, max);
			// level 2
			min = _mm512_min_epi32(L, H);
			max = _mm512_max_epi32(L, H);
			L = _mm512_mask_blend_epi32(kL2L3, min, max);
			H = _mm512_permutex2var_epi32(min, vL2L3, max);
			// level 3
			min = _mm512_min_epi32(L, H);
			max = _mm512_max_epi32(L, H);
			L = _mm512_mask_blend_epi32(kL3L4, min, max);
			H = _mm512_permutex2var_epi32(min, vL3L4, max);
			// level 4
			min = _mm512_min_epi32(L, H);
			max = _mm512_max_epi32(L, H);
			L = _mm512_mask_blend_epi32(kL4L5, min, max);
			H = _mm512_permutex2var_epi32(min, vL4L5, max);
			// level 5
			min = _mm512_min_epi32(L, H);
			max = _mm512_max_epi32(L, H);
			L = _mm512_permutex2var_epi32(min, vL5Out_L, max);
			H = _mm512_permutex2var_epi32(min, vL5Out_H, max);

#if !BLEND
			// deduplicate over block end, 1 2 3 ... 15 | 15 16 ...
			// get lowest and highest value from vector
			// no extract instruction to get scalar -> compressstore
			uint32_t tmp[16]; // only uses first 2 elements
			_mm512_mask_compressstoreu_epi32(tmp, 0x8001, L);
			uint32_t first = tmp[0];
			count -= (endofblock==first);
			endofblock = tmp[1];
			// deduplicate first 16 elements and compressstore them
			__m512i vconflict = _mm512_conflict_epi32(L);
			__mmask16 kconflict = _mm512_cmpeq_epi32_mask(vconflict, _mm512_setzero_epi32());
			_mm512_mask_compressstoreu_epi32(&result[count], kconflict, L);
			count += _mm_popcnt_u32(kconflict); // number of elements written
#else
			__m512i recon = _mm512_mask_blend_epi32(0x7fff, old, L);
			const __m512i circlic_shuffle = _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15);
			__m512i dedup = _mm512_permutexvar_epi32(circlic_shuffle, recon);
			__mmask16 kmask = _mm512_cmpneq_epi32_mask(dedup, L);
			_mm512_mask_compressstoreu_epi32(&result[count], kmask, L);
			count += _mm_popcnt_u32(kmask); // number of elements written
			// remember minimum for next iteration
			old = L;
#endif

			v_a = H;
			// compare first element of the next block in both lists
			a_nextfirst = list1[i_a+16];
			b_nextfirst = list2[i_b+16];
			// write minimum as above out to result
			// keep maximum and do the same steps as above with next block
			// next block from one list, which first element in new block is smaller
			i_a += (a_nextfirst <= b_nextfirst) * 16;
			i_b += (a_nextfirst > b_nextfirst) * 16;
			size_t index = (a_nextfirst <= b_nextfirst)? i_a: i_b;
			const uint32_t *base = (a_nextfirst <= b_nextfirst)? list1: list2;
			v_b = _mm512_load_epi32(&base[index]);
		}while(i_a < st_a && i_b < st_b);
		
		// v_a contains max vector from last comparison, v_b contains new, might be out of bounds
		// indices i_a and i_b correct, still need to handle v_a
		_mm512_store_epi32(maxtail, _mm512_permutexvar_epi32(vreverse, v_a));
#if BLEND
		uint32_t endofblock = _mm_extract_epi32(_mm512_extracti32x4_epi32(old, 3), 3);
#endif

		size_t mti=0;
		size_t mtsize = std::unique(maxtail, maxtail+16) - maxtail; // deduplicate tail
		if(a_nextfirst <= b_nextfirst){
			// endofblock needs to be considered too, for deduplication
			if(endofblock == std::min(maxtail[0],list1[i_a])) --count;
			// compare maxtail with list1
			while(mti < mtsize && i_a < size1){
				if(maxtail[mti] < list1[i_a]){
					result[count++] = maxtail[mti];
					mti++;
				}else if(maxtail[mti] > list1[i_a]){
					result[count++] = list1[i_a];
					i_a++;
				}else{
					result[count++] = maxtail[mti];
					mti++; i_a++;
				}
			}
			i_b += 16;
		}else{
			// endofblock needs to be considered too, for deduplication
			if(endofblock == std::min(maxtail[0],list2[i_b])) --count;
			// compare maxtail with list2
			while(mti < mtsize && i_b < size2){
				if(maxtail[mti] < list2[i_b]){
					result[count++] = maxtail[mti];
					mti++;
				}else if(maxtail[mti] > list2[i_b]){
					result[count++] = list2[i_b];
					i_b++;
				}else{
					result[count++] = maxtail[mti];
					mti++; i_b++;
				}
			}
			i_a += 16;
		}
		while(mti < mtsize){
			result[count++] = maxtail[mti++];
		}
	}

	// scalar tail
	count += union_scalar_branchless(list1+i_a, size1-i_a, list2+i_b, size2-i_b, result+count);

#endif
	return count;
}

#undef BLEND

#endif
