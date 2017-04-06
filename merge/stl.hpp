#ifndef MERGE_STL_HPP_
#define MERGE_STL_HPP_

#include <algorithm>


size_t merge_scalar_stl(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	uint32_t *endresult = std::merge(list1, list1+size1, list2, list2+size2, result);
	return endresult-result;
}

#endif
