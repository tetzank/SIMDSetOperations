#ifndef INTERSECTION_STL_HPP_
#define INTERSECTION_STL_HPP_

#include <algorithm>
#if __GNUC__ >= 5 && __GNUC__ < 8
#include <parallel/algorithm>
#endif


size_t intersect_scalar_stl(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	uint32_t *endresult = std::set_intersection(list1, list1+size1, list2, list2+size2, result);
	return endresult-result;
}

size_t intersect_scalar_stl_parallel(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
#if __GNUC__ >= 5 && __GNUC__ < 8
	uint32_t *endresult = __gnu_parallel::set_intersection((uint32_t*)list1, (uint32_t*)(list1+size1), (uint32_t*)list2, (uint32_t*)(list2+size2), result);
	return endresult-result;
#else
	return 0;
#endif
}


#endif
