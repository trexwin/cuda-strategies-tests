#include "./Measurements.cu"
#include "./Pragmas.cu"

__global__ void prefold2InlineKernel(int *input, int* output)
{
	__shared__ int sdata[THREADBLOCKSIZE];
	const unsigned int tx = threadIdx.x;
	const unsigned int bx = blockIdx.x;
	const unsigned int bdimx = blockDim.x;
	const size_t offset = bx * (bdimx * 2) + tx;
	
	// Read data, else in case of awkward size
	if(offset + bdimx < INPUTSIZE)
		sdata[tx] = input[offset] + input[offset + bdimx];
	else if(offset < INPUTSIZE)
		sdata[tx] = input[offset];
	else 
		sdata[tx] = 0;
	
	__syncthreads();
	
	// Folding
	FOR_BASIC_REDUCTION_X(sdata, tx, THREADBLOCKSIZE/2, 0)
	if(tx == 0)
		output[bx] = sdata[0];
}

void prefold2Inline(Measurements* measurements)
{
	const size_t inputByteSize = INPUTSIZE * sizeof(int);
	const size_t outputByteSize = (GRIDSIZE+1)/2 * sizeof(int);
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
		prefold2InlineKernel<<<(GRIDSIZE+1)/2, THREADBLOCKSIZE>>>(d_input, d_output); // +1 to avoid rounding down
		recordTime(measurements, measurements->gpuFoldTime, stopGpuTimer(measurements));
		cudaDeviceSynchronize();
		measurements->iterations++;
	}
	
	cudaMemcpy(h_output, d_output, outputByteSize, cudaMemcpyDeviceToHost);
	
	// Linear fold on CPU
	for(size_t i = 0; i < (GRIDSIZE+1)/2; ++i)
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
	char foldName[] = "Prefold_2Inline";
	Measurements* measurements = measurements_new((char*)&foldName);
	prefold2Inline(measurements);
	printMeasurements(measurements);
	
	free(measurements);
	
	return 0;
}
#endif
