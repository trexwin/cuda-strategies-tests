#include "./Measurements.cu"
#include "./Pragmas.cu"

__global__ void globalSyncKernel(const int *input, int* output, int *sync)
{
	__shared__ int sdata[THREADBLOCKSIZE];
	const unsigned int tx = threadIdx.x;
	const unsigned int bx = blockIdx.x; 
	const unsigned int bdimx = blockDim.x;
	const unsigned int gdimx = gridDim.x;
	size_t offset = bx * bdimx + tx;
	
	// Read data
	sdata[tx] = offset < INPUTSIZE ? input[offset] : 0;
	__syncthreads();
	
	// Folding
	FOR_BASIC_REDUCTION_X(sdata, tx, THREADBLOCKSIZE/2, 0)
	
	__shared__ bool isLastGroup;
	if(tx == 0) {
		output[bx] = sdata[0];
		__threadfence();
		isLastGroup = atomicAdd(sync, 1) == (gdimx-1);
	}
	__syncthreads();
	
	if(isLastGroup) {
		// Copy partial results into shared data
		offset = tx;
		sdata[tx] = 0;
		while(offset < gdimx) {
			sdata[tx] += output[offset];
			offset += bdimx;
		}
		__syncthreads();
		
		// Fold 
		FOR_BASIC_REDUCTION_X(sdata, tx, THREADBLOCKSIZE/2, 0)
		
		// Write final result
		if(tx == 0)
			output[0] = sdata[0];
	}
}

void globalSync(Measurements* measurements)
{
    const size_t inputByteSize = INPUTSIZE * sizeof(int);
	const size_t outputByteSize = GRIDSIZE * sizeof(int);
	int *h_input, *d_input, *d_output, *sync;
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
	cudaMalloc(&d_output, outputByteSize);
	cudaMalloc(&sync, sizeof(int));	
	cudaMemset(sync, 0, sizeof(int));
	
	// Fold operation
	for(int i = 0; i < ITERATIONS; ++i) {
		startGpuTimer(measurements);
        globalSyncKernel<<<GRIDSIZE, THREADBLOCKSIZE>>>(d_input, d_output, sync);
		recordTime(measurements, measurements->gpuFoldTime, stopGpuTimer(measurements));
		cudaDeviceSynchronize();
        cudaMemset(sync, 0, sizeof(int));
		cudaDeviceSynchronize();
		measurements->iterations++;
	}
    cudaMemcpy(&res, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    
	// Update final measurements
	measurements->success = measurements->success && res == trueRes;
	checkGpuErrors(measurements);
	
	// Free everything
	free(h_input);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(sync);
}

#ifndef HASMAIN
#define HASMAIN
int main(void)
{
	char foldName[] = "Global_Sync";
	Measurements* measurements = measurements_new((char*)&foldName);
	globalSync(measurements);
	printMeasurements(measurements);
	
	free(measurements);
	
	return 0;
}
#endif
