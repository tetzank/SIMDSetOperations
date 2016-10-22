#ifndef DIFFERENCE_NAIVE_HPP_
#define DIFFERENCE_NAIVE_HPP_

#include <cstring>


size_t difference_scalar(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t counter=0;
	const uint32_t *end1 = list1+size1, *end2 = list2+size2;
	while(list1 != end1){
		if(list2 == end2){
			memcpy(result+counter, list1, (end1-list1)*sizeof(uint32_t));
			return counter + (end1 - list1);
		}
		if(*list1 < *list2){
			result[counter++] = *list1;
			list1++;
		}else if(*list1 > *list2){
			list2++;
		}else{
			list1++; list2++;
		}
	}
	return counter;
}

#endif
