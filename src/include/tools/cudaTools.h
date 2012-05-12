#ifndef CUDA_TOOLS_H_
#define CUDA_TOOLS_H_

#include <string>

#include "builtin_types.h"
#include "cuda_runtime.h"

using std::string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

void handleError(cudaError_t error, const char *file, int line);
void print(const dim3& dim, string titre);
void print(const dim3& blockPerGrid,const  dim3& threadPerBlock);
void assertDimLegal(const dim3& threadPerGrid, const dim3& threadPerBlock);

/**
 * nTotal=#thread ideal
 *
 * Example 1
 * 	nTotal=256
 * 	nbBlock=128
 * 	threadPerBlock=2
 * Example 2
 * 	nTotal=257
 * 	nbBlock=128
 * 	threadPerBlock=3
 */
int threadPerBlock(int nTotal, int nbBlock);

#define HANDLE_ERROR( error ) (handleError( error, __FILE__, __LINE__ ))

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

