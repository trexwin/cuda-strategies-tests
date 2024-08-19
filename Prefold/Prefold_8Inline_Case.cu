#include "./Measurements.cu"
#include "./Pragmas.cu"

__global__ void prefold8InlineKernel(int *input, int* output)
{
	__shared__ int sdata[THREADBLOCKSIZE];
	const unsigned int tx = threadIdx.x;
	const unsigned int bx = blockIdx.x;
	const unsigned int bdimx = blockDim.x;
	const size_t offset = bx * 8 * bdimx + tx;
	
	// Read data, else in case of awkward size
	if(offset + 7 * bdimx < INPUTSIZE)
		sdata[tx] = input[offset] + 
					input[offset + bdimx] + 
					input[offset + 2 * bdimx] + 
					input[offset + 3 * bdimx] + 
					input[offset + 4 * bdimx] + 
					input[offset + 5 * bdimx] + 
					input[offset + 6 * bdimx] + 
					input[offset + 7 * bdimx];
	else if(offset + 6 * bdimx < INPUTSIZE)
		sdata[tx] = input[offset] + 
					input[offset + bdimx] + 
					input[offset + 2 * bdimx] + 
					input[offset + 3 * bdimx] + 
					input[offset + 4 * bdimx] + 
					input[offset + 5 * bdimx] + 
					input[offset + 6 * bdimx];
	else if(offset + 5 * bdimx < INPUTSIZE)
		sdata[tx] = input[offset] + 
					input[offset + bdimx] + 
					input[offset + 2 * bdimx] + 
					input[offset + 3 * bdimx] + 
					input[offset + 4 * bdimx] + 
					input[offset + 5 * bdimx];
	else if(offset + 4 * bdimx < INPUTSIZE)
		sdata[tx] = input[offset] + 
					input[offset + bdimx] + 
					input[offset + 2 * bdimx] + 
					input[offset + 3 * bdimx] + 
					input[offset + 4 * bdimx];
	else if(offset + 3 * bdimx < INPUTSIZE)
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

void prefold8Inline(Measurements* measurements)
{
	const size_t inputByteSize = INPUTSIZE * sizeof(int);
	const size_t outputByteSize = (GRIDSIZE+7)/8 * sizeof(int);
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
		prefold8InlineKernel<<<(GRIDSIZE+7)/8, THREADBLOCKSIZE>>>(d_input, d_output); // +7 to avoid rounding down
		recordTime(measurements, measurements->gpuFoldTime, stopGpuTimer(measurements));
		cudaDeviceSynchronize();
		measurements->iterations++;
	}
	
	cudaMemcpy(h_output, d_output, outputByteSize, cudaMemcpyDeviceToHost);
	
	// Linear fold on CPU
	for(size_t i = 0; i < (GRIDSIZE+7)/8; ++i)
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
	char foldName[] = "Prefold_8Inline";
	Measurements* measurements = measurements_new((char*)&foldName);
	prefold8Inline(measurements);
	printMeasurements(measurements);
	
	free(measurements);
	
	return 0;
}
#endif
