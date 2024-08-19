#define HASMAIN

#include <string>

#include "./Measurements.cu"
#include "./Base_Case.cu"

#include "./Strategy_For_Case.cu"
#include "./Strategy_1DUnrolled_Case.cu"
#include "./Strategy_2DUnrolled_Case.cu"
#include "./Strategy_1DNested_Case.cu"
#include "./Strategy_2DNested_Case.cu"

Measurements* testFold(const char* name, int iterations, void(*fold)(Measurements*))
{
	Measurements* measurements = measurements_new(name);
	fold(measurements);
	printMeasurements(measurements);
	
	return measurements;
}
 
int main(void) 
{
	// Initialise CUDA. https://stackoverflow.com/questions/15166799/any-particular-function-to-initialize-gpu-other-than-the-first-cudamalloc-call
	cudaFree(0);
	
	// Print the header if csv, otherwise environmental data
	#ifdef CSV
	printCSVHeader();
	#endif
	#ifndef CSV
	printParameters();
	#endif
	
	char baseName[] = "Base_Case";
	free(testFold((char*)&baseName, ITERATIONS, baseCase));
	
	//Fold strategies
	char strategyForName[] = "Strategy_For";
	free(testFold((char*)&strategyForName, ITERATIONS, strategyFor));
	char strategy1DUnrolledName[] = "Strategy_1DUnrolled";
	free(testFold((char*)&strategy1DUnrolledName, ITERATIONS, strategy1DUnrolled));
	char strategy2DUnrolledName[] = "Strategy_2DUnrolled";
	free(testFold((char*)&strategy2DUnrolledName, ITERATIONS, strategy2DUnrolled));
	char strategy1DNestedName[] = "Strategy_1DNested";
	free(testFold((char*)&strategy1DNestedName, ITERATIONS, strategy1DNested));
	char strategy2DNestedName[] = "Strategy_2DNested";
	free(testFold((char*)&strategy2DNestedName, ITERATIONS, strategy2DNested));

	return 0;
}

