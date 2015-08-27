#include <cstdio>
#include <algorithm>

#include "constants.h"


int main(int argc, char **argv){
	uint32_t *numbers = new uint32_t[poolSize];
	// create simple list
	for(size_t i=0; i<poolSize; ++i){
		numbers[i] = i;
	}
	uint32_t *list = new uint32_t[arraySize];
	std::random_device rng;
	FILE *fd = fopen("test.dat", "wb");

	for(size_t i=0; i<listCount; ++i){
		// random shuffle
		std::shuffle(numbers, numbers+poolSize, rng);
		// copy smaller list from big random list
		std::copy(numbers, numbers+arraySize, list);
		// sort list
		std::sort(list, list+arraySize);

		// write to file
		fwrite(list, 4, arraySize, fd);
		printf("done %3i/%i\n", i+1, listCount);
	}
	fclose(fd);

	delete[] numbers;
	delete[] list;

	return 0;
}
