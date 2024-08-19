#ifndef REDUCTION_PRAGMAS
#define REDUCTION_PRAGMAS

#define BASIC_REDUCTION_X(sdata, tx, step)		\
	if(tx < step)								\
		sdata[tx] += sdata[tx + step];			\
	__syncthreads();

#define FOR_BASIC_REDUCTION_X(sdata, tx, from, to)		\
	for(int i = from; i > to; i >>= 1) {		\
		BASIC_REDUCTION_X(sdata, tx, i)			\
	}

#define PART_NESTED_REDUCTION_X(sdata, tx, step)	\
	if(tx < step) {									\
		sdata[tx] += sdata[tx + step];				\
		__syncthreads();


#define BASIC_REDUCTION_Y(sdata, tx, ty, step)	\
	if(ty < step)								\
		sdata[ty][tx] += sdata[ty + step][tx];	\
	__syncthreads();
	
#define PART_NESTED_REDUCTION_Y(sdata, tx, ty, step)	\
	if(ty < step) {										\
		sdata[ty][tx] += sdata[ty + step][tx];			\
		__syncthreads();

	
#endif