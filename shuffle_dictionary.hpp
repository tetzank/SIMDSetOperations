#ifndef SHUFFLE_DICTIONARY_HPP_
#define SHUFFLE_DICTIONARY_HPP_

#include <cstring>
#include <immintrin.h>


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

#endif
