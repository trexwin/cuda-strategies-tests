#include "./Measurements.cu"
#include "./Pragmas.cu"

__global__ void warp1DSyncWarKernel(const int *input, int* output)
{
	__shared__ int sdata[THREADBLOCKSIZE];
	const unsigned int tx = threadIdx.x;
	const unsigned int bx = blockIdx.x; 
	const size_t offset = bx * blockDim.x + tx;
	
	// Read data
	sdata[tx] = offset < INPUTSIZE ? input[offset] : 0;
	__syncthreads();
	
	// Folding
	FOR_BASIC_REDUCTION_X(sdata, tx, THREADBLOCKSIZE/2, 32)

	if(tx < 32) {
        sdata[tx] += sdata[tx+32];
        __syncwarp ();
        sdata[tx] += sdata[tx+16];
        __syncwarp ();
        sdata[tx] += sdata[tx+8];
        __syncwarp ();
        sdata[tx] += sdata[tx+4];
        __syncwarp ();
        sdata[tx] += sdata[tx+2];
        __syncwarp ();
        sdata[tx] += sdata[tx+1];
		if(tx == 0)
			output[bx] = sdata[0];
	}
}

void warp1DSyncWar(Measurements* measurements)
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
		warp1DSyncWarKernel<<<GRIDSIZE, THREADBLOCKSIZE>>>(d_input, d_output);
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
	char foldName[] = "Warp_Syncwarp_1D";
	Measurements* measurements = measurements_new((char*)&foldName);	
	warp1DSyncWar(measurements);
	printMeasurements(measurements);
	
	free(measurements);
	
	return 0;
}
#endif
