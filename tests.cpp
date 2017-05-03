#include <cstdio>
#include <cstdint>

#include <array>
#include <vector>

#include "shuffle_dictionary.hpp"

#include "intersection/naive.hpp"
#include "intersection/stl.hpp"
#include "intersection/branchless.hpp"
#include "intersection/sse.hpp"
#include "intersection/avx.hpp"
#include "intersection/avx2.hpp"
#include "intersection/avx512.hpp"

#include "union/naive.hpp"
#include "union/stl.hpp"
#include "union/branchless.hpp"
#include "union/sse.hpp"
#include "union/avx512.hpp"

#include "difference/naive.hpp"
#include "difference/stl.hpp"
#include "difference/sse.hpp"
#include "difference/avx2.hpp"
#include "difference/avx512.hpp"

#include "merge/naive.hpp"
#include "merge/stl.hpp"
#include "merge/sse.hpp"
#include "merge/avx512.hpp"


bool equivalent(const uint32_t *list1, int size1, const uint32_t *list2, int size2){
	if(size1 != size2) return false;
	for(int i=0; i<size1; ++i){
		if(list1[i] != list2[i]) return false;
	}
	return true;
}

struct testcase{
	const char *name;
	uint32_t *list1, *list2;
	size_t size1, size2;
	uint32_t *res_intersection, *res_union, *res_difference, *res_merge;
	size_t size_intersection, size_union, size_difference, size_merge;

	testcase(
		const char *name,
		std::initializer_list<uint32_t> list1,
		std::initializer_list<uint32_t> list2,
		std::initializer_list<uint32_t> res_intersection,
		std::initializer_list<uint32_t> res_union,
		std::initializer_list<uint32_t> res_difference,
		std::initializer_list<uint32_t> res_merge
	) : name(name)
	{
		size1 = list1.size();
		this->list1 = (uint32_t*)aligned_alloc(64, size1*sizeof(uint32_t));
		std::copy(list1.begin(), list1.end(), this->list1);
		size2 = list2.size();
		this->list2 = (uint32_t*)aligned_alloc(64, size2*sizeof(uint32_t));
		std::copy(list2.begin(), list2.end(), this->list2);
		size_intersection = res_intersection.size();
		this->res_intersection = (uint32_t*)aligned_alloc(64, size_intersection*sizeof(uint32_t));
		std::copy(res_intersection.begin(), res_intersection.end(), this->res_intersection);
		size_union = res_union.size();
		this->res_union = (uint32_t*)aligned_alloc(64, size_union*sizeof(uint32_t));
		std::copy(res_union.begin(), res_union.end(), this->res_union);
		size_difference = res_difference.size();
		this->res_difference = (uint32_t*)aligned_alloc(64, size_difference*sizeof(uint32_t));
		std::copy(res_difference.begin(), res_difference.end(), this->res_difference);
		size_merge = res_merge.size();
		this->res_merge = (uint32_t*)aligned_alloc(64, size_merge*sizeof(uint32_t));
		std::copy(res_merge.begin(), res_merge.end(), this->res_merge);
	}
	// making sure nobody even tries as it would lead to use-after-free
	testcase(const testcase&) = delete;

	~testcase(){
		free(list1);
		free(list2);
		free(res_intersection);
		free(res_union);
		free(res_difference);
		free(res_merge);
	}
};

typedef size_t (*func_t)(const uint32_t*,size_t,const uint32_t*,size_t,uint32_t*);
void run(
	const testcase *tests, size_t tests_size,
	std::vector<std::pair<const char*,func_t>> f_intersection,
	std::vector<std::pair<const char*,func_t>> f_union,
	std::vector<std::pair<const char*,func_t>> f_difference,
	std::vector<std::pair<const char*,func_t>> f_merge
){
	//for(const auto &t : tests){
	for(size_t i=0; i<tests_size; ++i){
		const auto &t = tests[i];
		uint32_t *res = (uint32_t*)aligned_alloc(64, (t.size1+t.size2)*sizeof(uint32_t));
		for(const auto &f : f_intersection){
			size_t size_res = f.second(t.list1, t.size1, t.list2, t.size2, res);
			if(!equivalent(res, size_res, t.res_intersection, t.size_intersection)){
				//TODO
				printf("test \"%s\", intersection \"%s\" wrong\nlist1 : ", t.name, f.first);
				for(size_t i=0; i<t.size1; ++i) printf("%i, ", t.list1[i]);
				printf("\nlist2 : ");
				for(size_t i=0; i<t.size2; ++i) printf("%i, ", t.list2[i]);
				printf("\nresult: ");
				for(size_t i=0; i<size_res; ++i) printf("%i, ", res[i]);
				printf("\nexpect: ");
				for(size_t i=0; i<t.size_intersection; ++i) printf("%i, ", t.res_intersection[i]);
				printf("\n");
			}
		}
		for(const auto &f : f_union){
			size_t size_res = f.second(t.list1, t.size1, t.list2, t.size2, res);
			if(!equivalent(res, size_res, t.res_union, t.size_union)){
				//TODO
				printf("test \"%s\", union \"%s\" wrong\nlist1 : ", t.name, f.first);
				for(size_t i=0; i<t.size1; ++i) printf("%i, ", t.list1[i]);
				printf("\nlist2 : ");
				for(size_t i=0; i<t.size2; ++i) printf("%i, ", t.list2[i]);
				printf("\nresult: ");
				for(size_t i=0; i<size_res; ++i) printf("%i, ", res[i]);
				printf("\nexpect: ");
				for(size_t i=0; i<t.size_union; ++i) printf("%i, ", t.res_union[i]);
				printf("\n");
			}
		}
		for(const auto &f : f_difference){
			size_t size_res = f.second(t.list1, t.size1, t.list2, t.size2, res);
			if(!equivalent(res, size_res, t.res_difference, t.size_difference)){
				//TODO
				printf("test \"%s\", difference \"%s\" wrong\nlist1 : ", t.name, f.first);
				for(size_t i=0; i<t.size1; ++i) printf("%i, ", t.list1[i]);
				printf("\nlist2 : ");
				for(size_t i=0; i<t.size2; ++i) printf("%i, ", t.list2[i]);
				printf("\nresult: ");
				for(size_t i=0; i<size_res; ++i) printf("%i, ", res[i]);
				printf("\nexpect: ");
				for(size_t i=0; i<t.size_difference; ++i) printf("%i, ", t.res_difference[i]);
				printf("\n");
			}
		}
		for(const auto &f : f_merge){
			size_t size_res = f.second(t.list1, t.size1, t.list2, t.size2, res);
			if(!equivalent(res, size_res, t.res_merge, t.size_merge)){
				//TODO
				printf("test \"%s\", merge \"%s\" wrong\nlist1 : ", t.name, f.first);
				for(size_t i=0; i<t.size1; ++i) printf("%i, ", t.list1[i]);
				printf("\nlist2 : ");
				for(size_t i=0; i<t.size2; ++i) printf("%i, ", t.list2[i]);
				printf("\nresult: ");
				for(size_t i=0; i<size_res; ++i) printf("%i, ", res[i]);
				printf("\nexpect: ");
				for(size_t i=0; i<t.size_merge; ++i) printf("%i, ", t.res_merge[i]);
				printf("\n");
			}
		}
		free(res);
	}
}

int main(){
	const testcase tests[] = {
		{
			"equal lists",
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64}, // list1
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64}, // list2
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64}, // intersection result
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64}, // union result
			{},                                               // difference result
			{0,0,2,2,4,4,7,7, 11,11,13,13,23,23,32,32, 33,33,42,42,44,44,48,48, 53,53,55,55,60,60,64,64}, // merge result
		},{
			"completely different lists",
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64},
			{1,3,5,6, 10,12,27,31, 36,47,50,51, 52,66,77,88},
			{},
			{0,1,2,3,4,5,6,7, 10,11,12,13,23,27,31,32, 33,36,42,44,47,48,50,51, 52,53,55,60,64,66,77,88},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64},
			{0,1,2,3,4,5,6,7, 10,11,12,13,23,27,31,32, 33,36,42,44,47,48,50,51, 52,53,55,60,64,66,77,88},
		},{
			"no match in first",
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 77,78,79,80, 81,82,83,84, 87,88,89,99},
			{1,3,5,6, 10,12,27,31, 36,47,50,51, 52,66,77,88},
			{66,77,88},
			{0,1,2,3,4,5,6,7, 10,11,12,13,23,27,31,32, 33,36,42,44,47,48,50,51, 52,53,55,60,64, 66,67,68,69, 77,78,79,80, 81,82,83,84, 87,88,89,99},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 67,68,69, 78,79,80, 81,82,83,84, 87,89,99},
			{0,1,2,3,4,5,6,7, 10,11,12,13,23,27,31,32, 33,36,42,44,47,48,50,51, 52,53,55,60,64,66,66,67,68,69,77,77,78,79,80,81,82,83,84,87,88,88,89,99}
		},{
			"duplicate across vectors of different sizes",
			{0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,128},
			{1, 2, 4, 7, 9,10,13,14,17,18,19,22,23,27,29,30, 33,36,42,44,48,49,52,55,61,66,69,74,77,88,99,101},
			{1,  2,  4,  7,  9, 10, 13, 14, 17, 18, 19, 22, 23, 27, 29, 30},
			{0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 36, 42, 44, 48, 49, 52, 55, 61, 66, 69, 74, 77, 88, 99,101,128},
			{0,  3,  5,  6,  8, 11, 12, 15, 16, 20, 21, 24, 25, 26, 28,128},
			{0,1,1,2,2,3,4,4,5,6,7,7,8,9,9,10,10,11,12,13,13,14,14,15, 16,17,17,18,18,19,19,20,21,22,22,23,23,24,25,26,27,27,28,29,29,30,30, 33,36,42,44,48,49,52,55,61,66,69,74,77,88,99,101, 128}
		},{
			"one list very small",
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 77,78,79,80, 81,82,83,84, 87,88,89,99},
			{88},
			{88},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 77,78,79,80, 81,82,83,84, 87,88,89,99},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 77,78,79,80, 81,82,83,84, 87,89,99},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 77,78,79,80, 81,82,83,84, 87,88,88,89,99},
		},{
			"checking tail handling",
			{0, 3,14,32, 40,60,64,67, 72,75,80,86, 87,89,90,101, 109,111,116,133, 134,135,138,156, 158,172,178,180, 199},
			{6,10,12,21, 26,34,36,39, 41,43,46,49, 54,58,67, 81,  87, 93,105,113, 115,144,146,156, 159,160,164,171},
			{67, 87, 156},
			{0,3,6,10,12,14,21,26,32,34,36,39,40,41,43,46,49,54,58,60,64,67, 72,75,80,81,86,87, 89,90,93,101,105,109,111,113,115,116,133,134,135,138,144,146,156, 158,159,160,164,171,172,178,180, 199},
			{0, 3,14,32, 40,60,64, 72,75,80,86, 89,90,101, 109,111,116,133, 134,135,138, 158,172,178,180, 199},
			{0,3,6,10,12,14,21,26,32,34,36,39,40,41,43,46,49,54,58,60,64,67,67, 72,75,80,81,86,87,87, 89,90,93,101,105,109,111,113,115,116,133,134,135,138,144,146,156,156, 158,159,160,164,171,172,178,180, 199},
		},{
			"equal values for next vector",
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 70,71,72,73, 74,75,76,77, 78,79,80,81},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 70,71,72,73, 74,75,76,77, 78,79,80,81},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 70,71,72,73, 74,75,76,77, 78,79,80,81},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 70,71,72,73, 74,75,76,77, 78,79,80,81},
			{},
			{0,0,2,2,4,4,7,7, 11,11,13,13,23,23,32,32, 33,33,42,42,44,44,48,48, 53,53,55,55,60,60,64,64,
			66,66,67,67,68,68,69,69,70,70,71,71,72,72,73,73,74,74,75,75,76,76,77,77,78,78,79,79,80,80,81,81},
		},{
			"equal values for next vector, slightly different lists",
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 70,71,72,73, 74,75,76,77, 78,79,80,81},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 70,71,72,73, 74,75,76,77, 78,79,80,99},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 70,71,72,73, 74,75,76,77, 78,79,80},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 70,71,72,73, 74,75,76,77, 78,79,80,81,99},
			{81},
			{0,0,2,2,4,4,7,7, 11,11,13,13,23,23,32,32, 33,33,42,42,44,44,48,48, 53,53,55,55,60,60,64,64,
			66,66,67,67,68,68,69,69,70,70,71,71,72,72,73,73,74,74,75,75,76,76,77,77,78,78,79,79,80,80,81,99},
		}
	};
	constexpr int tests_size = sizeof(tests) / sizeof(testcase);

	#define FN(x) {#x, x}
	run(
		tests, tests_size,
		{
			FN(intersect_scalar),
			FN(intersect_scalar_stl),
			FN(intersect_scalar_branchless_c),
			FN(intersect_scalar_branchless),
			FN(intersect_vector_sse),
			FN(intersect_vector_sse_asm),
			FN(intersect_vector_avx),
#ifdef __AVX2__
			FN(intersect_vector_avx2),
			FN(intersect_vector_avx2_asm),
#endif
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
			FN(intersect_vector_avx512_conflict),
			FN(intersect_vector_avx512_conflict_asm)
#endif
		},
		{
			FN(union_scalar),
			FN(union_scalar_stl),
			FN(union_scalar_branchless),
			FN(union_vector_sse),
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
			FN(union_vector_avx512_bitonic),
			FN(union_vector_avx512_bitonic2)
#endif
		},
		{
			FN(difference_scalar),
			FN(difference_scalar_stl),
			FN(difference_vector_sse),
#ifdef __AVX2__
			FN(difference_vector_avx2),
#endif
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
			FN(difference_vector_avx512_conflict),
			FN(difference_vector_avx512_conflict_asm)
#endif
		},
		{
			FN(merge_scalar),
			FN(merge_scalar_stl),
			FN(merge_vector_sse),
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512DQ__)
			FN(merge_vector_avx512_bitonic2),
#endif
		}
	);

	return 0;
}
