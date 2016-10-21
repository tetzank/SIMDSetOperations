#ifndef INTERSECTION_BRANCHLESS_HPP_
#define INTERSECTION_BRANCHLESS_HPP_


size_t intersect_scalar_branchless_c(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	size_t counter=0, pos1=0, pos2=0;
	while(pos1 < size1 && pos2 < size2){
		uint32_t data1=list1[pos1], data2=list2[pos2];
		// a bit better, still not nearly as good as asm
		result[counter] = data1;
		counter += (data1 == data2);
		pos1 += (data1 <= data2);
		pos2 += (data1 >= data2);
	}
	return counter;
}

size_t intersect_scalar_branchless_c_count(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2){
	size_t counter=0, pos1=0, pos2=0;
	while(pos1 < size1 && pos2 < size2){
		uint32_t data1=list1[pos1], data2=list2[pos2];
		counter += (data1 == data2);
		pos1 += (data1 <= data2);
		pos2 += (data1 >= data2);
	}
	return counter;
}


size_t intersect_scalar_branchless(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	const uint32_t *end1 = list1+size1, *end2 = list2+size2, *endresult=result;
	asm(".intel_syntax noprefix;"
		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor rcx, rcx;"
	"1: "
		"cmp %[list1], %[end1];"  // list1 != end1
		"je 2f;"
		"cmp %[list2], %[end2];"  // list2 != end2
		"je 2f;"

		"mov r10d, [%q[list2]];"  // saved in r10d as value is only 4 byte wide
		"cmp [%q[list1]], r10d;"  // compare *list1 and *list2
		"setle al;"               // set al=1 if lower or equal
		"setge bl;"               // set bl=1 if greater or equal
		"sete  cl;"               // set cl=1 if equal, a bit quicker than: and rax, rbx;

		"mov [%q[endresult]], r10d;"  // always save, is overwritten when not equal

		"lea %q[list1], [%q[list1] + rax*4];"         // list1++, if lower or equal
		"lea %q[list2], [%q[list2] + rbx*4];"         // list2++, if greater or equal
		"lea %q[endresult], [%q[endresult] + rcx*4];" // result++, if equal

		"jmp 1b;"       // to loop head
	"2: "
		".att_syntax;"

		: [endresult]"=r"(endresult)
		: [list1]"r"(list1), [list2]"r"(list2), [end1]"r"(end1), [end2]"r"(end2),
			"0"(result)
		: "%rax","%rbx","%rcx", "%r10", "memory", "cc"
	);
	return endresult-result;
}

size_t intersect_scalar_branchless_count(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2){
	const uint32_t *end1 = list1+size1, *end2 = list2+size2;
	size_t count=0;
	asm(".intel_syntax noprefix;"
		"xor rax, rax;"
		"xor rbx, rbx;"
		"xor rcx, rcx;"
	"1: "
		"cmp %[list1], %[end1];"  // list1 != end1
		"je 2f;"
		"cmp %[list2], %[end2];"  // list2 != end2
		"je 2f;"

		"mov r10d, [%q[list2]];"  // saved in r10d as value is only 4 byte wide
		"cmp [%q[list1]], r10d;"  // compare *list1 and *list2
		"setle al;"               // set al=1 if lower or equal
		"setge bl;"               // set bl=1 if greater or equal
		"sete  cl;"               // set cl=1 if equal, a bit quicker than: and rax, rbx;

		"lea %q[list1], [%q[list1] + rax*4];" // list1++, if lower or equal
		"lea %q[list2], [%q[list2] + rbx*4];" // list2++, if greater or equal
		"lea %q[count], [%q[count] + rcx];"   // count++, if equal

		"jmp 1b;"       // to loop head
	"2: "
		".att_syntax;"

		: [count]"=r"(count)
		: [list1]"r"(list1), [list2]"r"(list2), [end1]"r"(end1), [end2]"r"(end2),
			"0"(count)
		: "%rax","%rbx","%rcx", "%r10"/*, "memory", "cc"*/
	);
	return count;
}

#endif
