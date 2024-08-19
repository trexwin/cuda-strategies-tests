#include "./Measurements.cu"
#include "./Pragmas.cu"

__global__ void strategy1DUnrolledKernel(const int *input, int* output)
{
	__shared__ int sdata[THREADBLOCKSIZE];
	const unsigned int tx = threadIdx.x;
	const unsigned int bx = blockIdx.x; 
	const size_t offset = bx * blockDim.x + tx;
	
	// Read data
	sdata[tx] = offset < INPUTSIZE ? input[offset] : 0;
	__syncthreads();
	
	// Folding
	#if THREADBLOCKSIZE >= 1024
	BASIC_REDUCTION_X(sdata, tx, 512)
	#endif
	#if THREADBLOCKSIZE >= 512
	BASIC_REDUCTION_X(sdata, tx, 256)
	#endif
	#if THREADBLOCKSIZE >= 256
	BASIC_REDUCTION_X(sdata, tx, 128)
	#endif
	#if THREADBLOCKSIZE >= 128
	BASIC_REDUCTION_X(sdata, tx, 64)
	#endif
	#if THREADBLOCKSIZE >= 64
	BASIC_REDUCTION_X(sdata, tx, 32)
	#endif
	#if THREADBLOCKSIZE >= 32
	BASIC_REDUCTION_X(sdata, tx, 16)
	#endif
	#if THREADBLOCKSIZE >= 16
	BASIC_REDUCTION_X(sdata, tx, 8)
	#endif
	#if THREADBLOCKSIZE >= 8
	BASIC_REDUCTION_X(sdata, tx, 4)
	#endif
	#if THREADBLOCKSIZE >= 4
	BASIC_REDUCTION_X(sdata, tx, 2)
	#endif
	#if THREADBLOCKSIZE >= 2
	BASIC_REDUCTION_X(sdata, tx, 1)
	#endif
	
	if(tx == 0)
		output[bx] = sdata[0];
}

void strategy1DUnrolled(Measurements* measurements)
{
    const size_t inputByteSize = INPUTSIZE * sizeof(int);
	const size_t outputByteSize = GRIDSIZE * sizeof(int);
	int *h_input, *d_input, *h_output, *d_output;
	int res = 0, trueRes = 0;
	
	// Initialise input arrays with 1s
	h_input = (int*)mallocMeasurements(inputByteSize, measurements);
	for(size_t i = 0; i < INPUTSIZE; ++i) {
		h_input[i] = i; 
		trueRes += i; 
	}
	
	cudaMalloc(&d_input, INPUTSIZE * sizeof(int));
	cudaMemcpy(d_input, h_input, inputByteSize, cudaMemcpyHostToDevice);
	
	// Initialise output arrays
	h_output = (int*)mallocMeasurements(outputByteSize, measurements);
	cudaMalloc(&d_output, outputByteSize);
	
	// Fold operation
	for(int i = 0; i < ITERATIONS; ++i) {
		startGpuTimer(measurements);
		strategy1DUnrolledKernel<<<GRIDSIZE, THREADBLOCKSIZE>>>(d_input, d_output);
		recordTime(measurements, measurements->gpuFoldTime, stopGpuTimer(measurements));
		cudaDeviceSynchronize();
		measurements->iterations++;
	}
	cudaMemcpy(h_output, d_output, outputByteSize, cudaMemcpyDeviceToHost);
	
	// Linear fold on CPU
	for(size_t i = 0; i < GRIDSIZE; ++i)
		res += h_output[i];
	
	// Update final measurements
	measurements->success = measurements->success && res == trueRes;
	checkGpuErrors(measurements);
	
	// Free everything
	free(h_input);
	free(h_output);
	cudaFree(d_input);
	cudaFree(d_output);
}

#ifndef HASMAIN
#define HASMAIN
int main(void)
{
	char foldName[] = "Strategy_1DUnrolled";
	Measurements* measurements = measurements_new((char*)&foldName);
	strategy1DUnrolled(measurements);
	printMeasurements(measurements);
	
	free(measurements);
	
	return 0;
}
#endif
