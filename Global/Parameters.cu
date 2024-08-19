// Environmental parameters
#ifndef INPUTSIZE
#define INPUTSIZE 1000000
#endif

#ifndef THREADBLOCKSIZE
#define THREADBLOCKSIZE 1024
#endif

#ifndef ITERATIONS
#define ITERATIONS 10
#endif

// If we have a remainder, we add 1 to the grid to add some padding
#define REMAIN (INPUTSIZE % THREADBLOCKSIZE)
#if REMAIN == 0
#define GRIDSIZE (INPUTSIZE / THREADBLOCKSIZE)
#endif
#if REMAIN != 0
#define GRIDSIZE (INPUTSIZE / THREADBLOCKSIZE + 1)
#endif


// Printing parameter code
#ifndef PARAMETERS
#define PARAMETERS

#include <stdio.h>

void printParameters() {
	printf("_Parameters_\n");
	printf("Input size: %zu\n",			(size_t)INPUTSIZE);
	printf("Thread block size: %i\n",	THREADBLOCKSIZE);
	printf("Grid size: %zu\n",			(size_t)GRIDSIZE);
	printf("Remainder: %zu\n-----\n",	(size_t)REMAIN);
}
#endif