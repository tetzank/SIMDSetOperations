#ifndef SHUFFLE_DICTIONARY_HPP_
#define SHUFFLE_DICTIONARY_HPP_

#include <cstring>
#include <immintrin.h>


// C++17 std::array has constexpr operator[], use our own for now
template<typename T, int N>
struct constarray{
	T elems[N];
	constexpr       T &operator[](size_t i)       { return elems[i]; }
	constexpr const T &operator[](size_t i) const { return elems[i]; }
};

static constexpr constarray<uint8_t,16*16> prepare_shuffling_dictionary(){
	constarray<uint8_t,16*16> arr = {0xff};
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
static const constexpr __m128i *shuffle_mask = (__m128i*)shuffle_mask_arr.elems;


static constexpr constarray<uint32_t,256*8> prepare_shuffling_dictionary_avx(){
	constarray<uint32_t,256*8> arr = {};
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
static const constexpr uint32_t *shuffle_mask_avx = shuffle_mask_avx_arr.elems;


#endif
