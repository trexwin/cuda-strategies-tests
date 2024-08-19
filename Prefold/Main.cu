#define HASMAIN

#include <string>

#include "./Measurements.cu"
#include "./Base_Case.cu"

#include "./Prefold_2Inline_Case.cu"
#include "./Prefold_4Inline_Case.cu"
#include "./Prefold_8Inline_Case.cu"
#include "./Prefold_Loop_Case.cu"

#define PREFOLD_LOOP(i)																		\
	{																						\
		std::string nameStart = {"Prefold_0"};												\
		std::string nameMid = {i > 999 ? "" : (i > 99 ? "0" : (i > 9 ? "00" : "000"))}; 	\
		std::string nameEnd = {"Loop"};														\
		std::string prefoldLoopName = nameStart + nameMid + std::to_string(i) + nameEnd;	\
		free(testFold(prefoldLoopName.c_str(), ITERATIONS, prefoldLoop<i>));				\
	}


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
	
	//Prefold cases
	char prefold2InlineName[] = "Prefold_2Inline";
	free(testFold((char*)&prefold2InlineName, ITERATIONS, prefold2Inline));
	char prefold4InlineName[] = "Prefold_4Inline";
	free(testFold((char*)&prefold4InlineName, ITERATIONS, prefold4Inline));
	char prefold8InlineName[] = "Prefold_8Inline";
	free(testFold((char*)&prefold8InlineName, ITERATIONS, prefold8Inline));
	
	PREFOLD_LOOP(2);
	PREFOLD_LOOP(4);
	PREFOLD_LOOP(8);
	PREFOLD_LOOP(16);
	PREFOLD_LOOP(32);
	PREFOLD_LOOP(64);
	PREFOLD_LOOP(128);
	PREFOLD_LOOP(256);
	return 0;
}

