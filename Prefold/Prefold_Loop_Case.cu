#include "./Measurements.cu"
#include "./Pragmas.cu"

__global__ void prefoldLoopKernel(int *input, int* output)
{
	__shared__ int sdata[THREADBLOCKSIZE];
	const unsigned int tx = threadIdx.x;
	const unsigned int bx = blockIdx.x;
	const unsigned int bdimx = blockDim.x;
	const unsigned int gdimx = gridDim.x;
	size_t offset = bx * bdimx + tx;
	const size_t gridSize = bdimx * gdimx;
	
	// Read data
	sdata[tx] = 0;
	while(offset < INPUTSIZE) {
		sdata[tx] += input[offset];
		offset += gridSize;
	}
	__syncthreads();
	
	// Folding
	FOR_BASIC_REDUCTION_X(sdata, tx, THREADBLOCKSIZE/2, 0)
	
	if(tx == 0)
		output[bx] = sdata[0];
}

template <unsigned int prefold>
void prefoldLoop(Measurements* measurements)
{
	const size_t inputByteSize = INPUTSIZE * sizeof(int);
	const size_t outputByteSize = (GRIDSIZE+(prefold-1))/prefold * sizeof(int);
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
		prefoldLoopKernel<<<(GRIDSIZE+(prefold-1))/prefold, THREADBLOCKSIZE>>>(d_input, d_output); // +x so same as inline
		recordTime(measurements, measurements->gpuFoldTime, stopGpuTimer(measurements));
		cudaDeviceSynchronize();
		measurements->iterations++;
	}
	cudaMemcpy(h_output, d_output, outputByteSize, cudaMemcpyDeviceToHost);
	
	// Linear fold on CPU
	for(size_t i = 0; i < (GRIDSIZE+(prefold-1))/prefold; ++i)
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
	char foldName[] = "Prefold_2Loop";
	Measurements* measurements = measurements_new((char*)&foldName);
	prefoldLoop<2>(measurements);
	printMeasurements(measurements);
	
	free(measurements);
	
	return 0;
}
#endif
