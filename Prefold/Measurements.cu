#ifndef MEASUREMENTS
#define MEASUREMENTS

#include <stdio.h>
#include <cuda_profiler_api.h>
#include "./Parameters.cu"

struct Measurements
{
	const char* name;
	bool success;
	unsigned int iterations;
    double gpuFoldTime[ITERATIONS];
	
	float cpuTimer, gpuTimer;
	cudaEvent_t gpuStart, gpuStop;
};

// Constructor
Measurements* measurements_new(const char* name) {
	Measurements* obj = (Measurements*)calloc(1, sizeof(Measurements));
	obj->name = name;
	obj->success = true;
	cudaEventCreate(&obj->gpuStart);
	cudaEventCreate(&obj->gpuStop);
	return obj;
}

// Error checking
void* mallocMeasurements(size_t size, Measurements* obj) {
	void* ptr = malloc(size);
	if(ptr == NULL)
		obj->success = false;
	return ptr;
}

void checkGpuErrors(Measurements* obj) {
	cudaError_t err{cudaGetLastError()};
	if(err != cudaSuccess)
		obj->success = false;
}

// Timers
void startCpuTimer(Measurements* obj) {
	obj->cpuTimer = clock();
}

double stopCpuTimer(Measurements* obj) {
	return (clock() - obj->cpuTimer) / CLOCKS_PER_SEC;
}

void startGpuTimer(Measurements* obj) {
	cudaEventRecord(obj->gpuStart);
}

double stopGpuTimer(Measurements* obj) {
	cudaEventRecord(obj->gpuStop);
	cudaEventSynchronize(obj->gpuStop);
	cudaEventElapsedTime(&obj->gpuTimer, obj->gpuStart, obj->gpuStop);
	return ((double)obj->gpuTimer)/1000;
}

// Administrative
void recordTime(Measurements* obj, double* arr, double time) {
	arr[obj->iterations] = time;
}

// Calculations
double calculateAverage(Measurements* obj, double* arr) {
	double total = 0;
	for(size_t i = 0; i < obj->iterations; ++i)
		total += arr[i];
	return total/obj->iterations;
}

double calculateStandardDeviation(Measurements* obj, double* arr, double average) {
	double total = 0;
	for(size_t i = 0; i < obj->iterations; ++i)
		total += pow(arr[i] - average, 2);
	return sqrt(total/obj->iterations);
}

double calculateStandardError(Measurements* obj, double stDeviation) {
	return stDeviation/sqrt(obj->iterations);
}

// Print
#ifdef CSV
void printCSVHeader(){
	printf("%s, %s \n",
		"Name, Input Size, Thread block Size, Iterations, GPU Fold", // 7
		"StDevGPU, StErrGPU"); // 2; // 3
}
#endif


void printMeasurements(Measurements* obj) {
	double 	gpuAvg = calculateAverage(obj, obj->gpuFoldTime),
			gpuStD = calculateStandardDeviation(obj, obj->gpuFoldTime, gpuAvg);
	
	#ifdef CSV
	// Mark wrong results with an X
	printf(obj->success ? "%s, " : "X_%s, ", obj->name);
	printf("%zu, ",	(size_t)INPUTSIZE);
	printf("%i, ", 	THREADBLOCKSIZE);
	
	//Data
	printf("%i, ", obj->iterations);
	// GPU data
	printf("%.10f, %.10f, %.10f\n", gpuAvg, gpuStD, calculateStandardError(obj, gpuStD));
	#endif
	
	#ifndef CSV
	printf("_%s_Statistics_\n", obj->name);
	printf("Iterations: %u\n", obj->iterations);
	printf(obj->success ? "Results are correct" : "Results are wrong");
	printf("GPU Fold: %f\n", gpuAvg);
	
	printf("Standard deviations\n");
	printf("GPU Fold: %f\n", gpuStD);
	#endif
}
#endif