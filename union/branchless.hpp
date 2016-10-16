#ifndef UNION_BRANCHLESS_HPP_
#define UNION_BRANCHLESS_HPP_



size_t union_scalar_branchless(const uint32_t *list1, size_t size1, const uint32_t *list2, size_t size2, uint32_t *result){
	const uint32_t *end1 = list1+size1, *end2 = list2+size2;
	uint32_t *endresult=result;
	asm(".intel_syntax noprefix;"
		"xor rax, rax;"
		"xor rbx, rbx;"
	"1: "
		"cmp %[list1], %[end1];"  // list1 != end1
		"je 3f;"
		"cmp %[list2], %[end2];"  // list2 != end2
		"je 2f;"

		"mov r11d, [%q[list1]];"  // move both to registers as cmov only works on registers
		"mov r10d, [%q[list2]];"  // saved in r10d as value is only 4 byte wide

		"cmp r11d, r10d;"   // compare *list1 and *list2
		"setbe al;"         // set al=1 if lower or equal
		"setae bl;"         // set bl=1 if greater or equal

		"cmovb r10d, r11d;" // save *list1 instead of *list2, if lower
		"mov [%q[endresult]], r10d;"  // always save

		"lea %q[list1], [%q[list1] + rax*4];"     // list1++, if lower or equal
		"lea %q[list2], [%q[list2] + rbx*4];"     // list2++, if greater or equal
		"lea %q[endresult], [%q[endresult] + 4];" // result++, always

		"jmp 1b;"       // to loop head
// 		"cmp %1, %3;"  // list1 != end1
// 		"je 2f;"
// 		"cmp %2, %4;"  // list2 != end2
// 		"jne 1b;"
	"2: "
		// inline memcpy
		//"cld;"
		// rsi - list1, rdi - endresult, rcx - end1
		"sub %[end1], %[list1];" //
		"rep movsb;"             // copy rest of list1
	"3: "
		"mov rcx, %[end2];"      // overwrites rcx which contained end1
		"mov rsi, %[list2];"     // overwrites rsi which contained list1
		"sub rcx, %[list2];"     //
		"rep movsb;"             // copy rest of list2
		".att_syntax;"

		: [endresult]"=D"(endresult), [list1]"=S"(list1), [list2]"=r"(list2)
		: [end1]"c"(end1), [end2]"r"(end2),
			"0"(endresult), "1"(list1), "2"(list2)
		: "%rax","%rbx", "%r10","%r11", "memory", "cc"
	);
	// copy rest, can't be the same
	//memcpy(endresult, list1, (end1-list1)*sizeof(uint32_t));
	//endresult += end1 - list1;
	//memcpy(endresult, list2, (end2-list2)*sizeof(uint32_t));
	//endresult += end2 - list2;

	return endresult-result;
}


#endif
