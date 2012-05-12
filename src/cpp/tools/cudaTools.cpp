#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "cudaTools.h"
#include "deviceTools.h"

using std::string;
using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int threadPerBlock(int nTotal, int nbBlock)
    {
    //return ceil(nTotal / (double) nbBlock);

    // mieux

    return (nTotal + (nbBlock - 1)) / nbBlock;
    }

void print(const dim3& dim, string titre)
    {
    cout << titre << "(" << dim.x << "," << dim.y << "," << dim.z << ")" << endl;
    }

void print(const dim3& blockPerGrid, const dim3& threadPerBlock)
    {
    print(blockPerGrid, "BlockPerGrid");
    print(threadPerBlock, "ThreadPerBlock");
    }

/**
 * of activeDevice!
 */
void assertDimLegal(const dim3& threadPerGrid, const dim3& threadPerBlock)
    {
    int deviceId;
    HANDLE_ERROR(cudaGetDevice(&deviceId));

    // grid
	{
	dim3 dimGridMax = getMaxGridDim(deviceId);

	assert(threadPerGrid.x <= dimGridMax.x);
	assert(threadPerGrid.y <= dimGridMax.y);
	assert(threadPerGrid.z <= dimGridMax.z);
	}

    // block
	{
	dim3 dimBlockMax = getMaxBlockDim(deviceId);

	assert(threadPerBlock.x <= dimBlockMax.x);
	assert(threadPerBlock.y <= dimBlockMax.y);
	assert(threadPerBlock.z <= dimBlockMax.z);
	}
    }

void handleError(cudaError_t error, const char *file, int line)
    {
    if (error != cudaSuccess)
	{
	cout << "[CUDA ERROR ] : " << cudaGetErrorString(error) << " in " << file << " at ligne " << line << endl;
	exit( EXIT_FAILURE);
	}
    }



/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

