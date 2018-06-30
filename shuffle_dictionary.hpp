#ifndef SHUFFLE_DICTIONARY_HPP_
#define SHUFFLE_DICTIONARY_HPP_

#include <cstring>
#include <array>
#include <immintrin.h>


#define IACA_START_ASM "mov ebx, 111;.byte 0x64, 0x67, 0x90;"
#define IACA_END_ASM   "mov ebx, 222;.byte 0x64, 0x67, 0x90;"


static constexpr std::array<uint8_t,16*16> prepare_shuffling_dictionary(){
	std::array<uint8_t,16*16> arr = {0xff};
	int size=0;
	for(int i=0; i<16; ++i){
		int counter=0;
		for(int j=0; j<4; ++j){
			if(i & (1 << j)){
				arr[size+counter*4  ] = 4*j;
				arr[size+counter*4+1] = 4*j + 1;
				arr[size+counter*4+2] = 4*j + 2;
				arr[size+counter*4+3] = 4*j + 3;
				++counter;
			}
		}
		size += 16;
	}
	return arr;
}
static const constexpr auto shuffle_mask_arr = prepare_shuffling_dictionary();
static const /*constexpr*/ __m128i *shuffle_mask = (__m128i*)shuffle_mask_arr.data();


static constexpr std::array<uint32_t,256*8> prepare_shuffling_dictionary_avx(){
	std::array<uint32_t,256*8> arr = {};
	for(int i=0; i<256; ++i){
		int count=0, rest=7;
		for(int b=0; b<8; ++b){
			if(i & (1 << b)){
				// n index at pos p - move nth element to pos p
				arr[i*8 + count] = b; // move all set bits to beginning
				++count;
			}else{
				arr[i*8 + rest] = b; // move rest at the end
				--rest;
			}
		}
	}
	return arr;
}
static const constexpr auto shuffle_mask_avx_arr = prepare_shuffling_dictionary_avx();
static const /*constexpr*/ uint32_t *shuffle_mask_avx = shuffle_mask_avx_arr.data();


#endif
