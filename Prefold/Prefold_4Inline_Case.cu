#include "./Measurements.cu"
#include "./Pragmas.cu"

__global__ void prefold4InlineKernel(int *input, int* output)
{
	__shared__ int sdata[THREADBLOCKSIZE];
	const unsigned int tx = threadIdx.x;
	const unsigned int bx = blockIdx.x;
	const unsigned int bdimx = blockDim.x;
	const size_t offset = bx * 4 * bdimx + tx;
	
	// Read data, else in case of awkward size
	if(offset + 3 * bdimx < INPUTSIZE)
		sdata[tx] = input[offset] + 
					input[offset + bdimx] + 
					input[offset + 2 * bdimx] + 
					input[offset + 3 * bdimx];
	else if(offset + 2 * bdimx < INPUTSIZE)
		sdata[tx] = input[offset] + 
					input[offset + bdimx] + 
					input[offset + 2 * bdimx];
	else if(offset + bdimx < INPUTSIZE)
		sdata[tx] = input[offset] + 
					input[offset + bdimx];
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

void prefold4Inline(Measurements* measurements)
{
	const size_t inputByteSize = INPUTSIZE * sizeof(int);
	const size_t outputByteSize = (GRIDSIZE+3)/4 * sizeof(int);
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
		prefold4InlineKernel<<<(GRIDSIZE+3)/4, THREADBLOCKSIZE>>>(d_input, d_output); // +3 to avoid rounding down
		recordTime(measurements, measurements->gpuFoldTime, stopGpuTimer(measurements));
		cudaDeviceSynchronize();
		measurements->iterations++;
	}
	
	cudaMemcpy(h_output, d_output, outputByteSize, cudaMemcpyDeviceToHost);
	
	// Linear fold on CPU
	for(size_t i = 0; i < (GRIDSIZE+3)/4; ++i)
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
	char foldName[] = "Prefold_4Inline";
	Measurements* measurements = measurements_new((char*)&foldName);
	prefold4Inline(measurements);
	printMeasurements(measurements);
	
	free(measurements);
	
	return 0;
}
#endif
