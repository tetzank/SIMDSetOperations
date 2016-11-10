#include <cstdio>
#include <cstdint>

#include <array>
#include <vector>

#include "shuffle_dictionary.hpp"

#include "intersection/sse.hpp"
#include "intersection/avx.hpp"
#include "intersection/avx2.hpp"

#include "union/sse.hpp"

#include "difference/sse.hpp"
#include "difference/avx2.hpp"


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
	uint32_t *res_intersection, *res_union, *res_difference;
	size_t size_intersection, size_union, size_difference;

	testcase(
		const char *name,
		std::initializer_list<uint32_t> list1,
		std::initializer_list<uint32_t> list2,
		std::initializer_list<uint32_t> res_intersection,
		std::initializer_list<uint32_t> res_union,
		std::initializer_list<uint32_t> res_difference
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
	}
	// making sure nobody even tries as it would lead to use-after-free
	testcase(const testcase&) = delete;

	~testcase(){
		free(list1);
		free(list2);
		free(res_intersection);
		free(res_union);
		free(res_difference);
	}
};

typedef size_t (*func_t)(const uint32_t*,size_t,const uint32_t*,size_t,uint32_t*);
void run(
	const testcase *tests, size_t tests_size,
	std::vector<std::pair<const char*,func_t>> f_intersection,
	std::vector<std::pair<const char*,func_t>> f_union,
	std::vector<std::pair<const char*,func_t>> f_difference
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
		free(res);
	}
}

int main(){
	prepare_shuffling_dictionary();
	prepare_shuffling_dictionary_avx();

	const testcase tests[] = {
		{
			"equal lists",
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64}, // list1
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64}, // list2
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64}, // intersection result
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64}, // union result
			{}  // difference result
		},{
			"completely different lists",
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64},
			{1,3,5,6, 10,12,27,31, 36,47,50,51, 52,66,77,88},
			{},
			{0,1,2,3,4,5,6,7, 10,11,12,13,23,27,31,32, 33,36,42,44,47,48,50,51, 52,53,55,60,64,66,77,88},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64}
		},{
			"no match in first",
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 66,67,68,69, 77,78,79,80, 81,82,83,84, 87,88,89,99},
			{1,3,5,6, 10,12,27,31, 36,47,50,51, 52,66,77,88},
			{66,77,88},
			{0,1,2,3,4,5,6,7, 10,11,12,13,23,27,31,32, 33,36,42,44,47,48,50,51, 52,53,55,60,64, 66,67,68,69, 77,78,79,80, 81,82,83,84, 87,88,89,99},
			{0,2,4,7, 11,13,23,32, 33,42,44,48, 53,55,60,64, 67,68,69, 78,79,80, 81,82,83,84, 87,89,99}
		}
	};
	constexpr int tests_size = sizeof(tests) / sizeof(testcase);

	#define FN(x) {#x, x}
	run(
		tests, tests_size,
		{
			FN(intersect_vector_sse),
			FN(intersect_vector_sse_asm),
			FN(intersect_vector_avx),
#ifdef __AVX2__
			FN(intersect_vector_avx2),
			FN(intersect_vector_avx2_asm)
#endif
		},
		{
			FN(union_vector_sse)
		},
		{
			FN(difference_vector_sse),
#ifdef __AVX2__
			FN(difference_vector_avx2),
#endif
		}
	);

	free(shuffle_mask_avx);
	return 0;
}
