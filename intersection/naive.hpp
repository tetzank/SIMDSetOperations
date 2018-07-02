#ifndef INTERSECTION_NAIVE_HPP_
#define INTERSECTION_NAIVE_HPP_

#if IACA
#include <iacaMarks.h>
#endif


size_t intersect_scalar(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t counter=0;
	const uint32_t *end1 = list1+size1, *end2 = list2+size2;
	// hard to get only the loop instructions, now only a tiny check at the top wrong
#if IACA_INTERSECT_SCALAR
		IACA_START
#endif
	while(list1 != end1 && list2 != end2){
		if(*list1 < *list2){
			list1++;
		}else if(*list1 > *list2){
			list2++;
		}else{
			result[counter++] = *list1;
			list1++; list2++;
		}
	}
#if IACA_INTERSECT_SCALAR
	IACA_END
#endif
	return counter;
}

size_t intersect_scalar_count(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2){
	size_t counter=0;
	const uint32_t *end1 = list1+size1, *end2 = list2+size2;
#if IACA_INTERSECT_SCALAR_COUNT
		IACA_START
#endif
	while(list1 != end1 && list2 != end2){
		if(*list1 < *list2){
			list1++;
		}else if(*list1 > *list2){
			list2++;
		}else{
			counter++;
			list1++; list2++;
		}
	}
#if IACA_INTERSECT_SCALAR_COUNT
	IACA_END
#endif
	return counter;
}

size_t intersect_scalar_index(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t counter=0;
	const uint32_t *end1 = list1+size1, *end2 = list2+size2;
	while(list1 != end1 && list2 != end2){
		if(*list1 < *list2){
			list1++;
		}else if(*list1 > *list2){
			list2++;
		}else{
			result[counter] = counter;
			++counter;
			list1++; list2++;
		}
	}
	return counter;
}

#endif
