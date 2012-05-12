#include <iostream>
#include "deviceTools.h"
#include "cudaTools.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

void cout01(int isTrue);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int nbDeviceDetect(void)
    {
    int nbDevice;
    HANDLE_ERROR(cudaGetDeviceCount(&nbDevice));

    return nbDevice;
    }

/**
 * idDevice in [0,nbDevice-1]
 */
cudaDeviceProp deviceProp(int idDevice)
    {
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, idDevice));
    return prop;
    }

/**
 * of device actif
 */
cudaDeviceProp deviceProp(void)
    {
    return deviceProp(getIdCurrentDevive());
    }

/**
 * idDevice in [0,nbDevice-1]
 */
void print(const cudaDeviceProp& ptrProp, int idDevice)
    {
    cout << endl;
    cout << "Device Info: " << endl;
    cout << "===========================================" << endl;
    cout << "Device id                : " << idDevice << endl;
    cout << "Name                     : " << ptrProp.name << endl;
    cout << "GPU capability           : " << ptrProp.major << "." << ptrProp.minor << "" << endl;
    cout << "Clock rate               : " << ptrProp.clockRate / 1000 << " MHZ" << endl;
    cout << "GPU Integrated in Chipset: ";
    cout01(ptrProp.integrated);
    cout << "ComputeMode              : " << ptrProp.computeMode << endl;

    cout << endl;
    cout << "Kernel : " << endl;
    cout << "Limit execution (timeout): ";
    cout01(ptrProp.kernelExecTimeoutEnabled);
    cout << "Concurrent copy/kernel   : ";
    cout01(ptrProp.deviceOverlap);
    cout << "Concurrent Kernels       : ";
    cout01(ptrProp.concurrentKernels);

    cout << endl;
    cout << "Memory : " << endl;
    cout << "Global Memory            : " << ptrProp.totalGlobalMem / 1024 / 1024 << " MB" << endl;
    cout << "Constant Memory          : " << ptrProp.totalConstMem / 1024 << " KB" << endl;
    cout << "Texture1D max size       : (" << ptrProp.maxTexture1D << ")" << endl;
    cout << "Texture2D max size       : (" << ptrProp.maxTexture2D[0] << "," << ptrProp.maxTexture2D[1] << ")" << endl;
    cout << "Texture3D max size       : (" << ptrProp.maxTexture3D[0] << "," << ptrProp.maxTexture3D[1] << "," << ptrProp.maxTexture3D[2] << ")" << endl;
    //cout << "Texture2D Array max Size : (" << ptrProp.maxTexture2DArray[0] << "," << ptrProp.maxTexture2DArray[1] << "," << ptrProp.maxTexture2DArray[2] << ")"<< endl;
    cout << "Texture Alignment        : " << ptrProp.textureAlignment << " B" << endl;
    cout << "Max mem pitch            : " << ptrProp.memPitch << endl;
    cout << "GPU can map Host memory  : ";
    cout01(ptrProp.canMapHostMemory);

    cout << endl;
    cout << "Multiprocesseur(MP)      : " << endl;
    cout << "MP count                 : " << ptrProp.multiProcessorCount << endl;
    cout << "Shared memory per block  : " << ptrProp.sharedMemPerBlock / 1024 << " KB " << endl;
    cout << "Register memory per block: " << ptrProp.regsPerBlock / 1024 << " KB " << endl;
    cout << "Max threads per block    : " << ptrProp.maxThreadsPerBlock << endl;
    cout << "Max thread dim           : (" << ptrProp.maxThreadsDim[0] << "," << ptrProp.maxThreadsDim[1] << "," << ptrProp.maxThreadsDim[2] << ")" << endl;
    cout << "Max grid dim             : (" << ptrProp.maxGridSize[0] << "," << ptrProp.maxGridSize[1] << "," << ptrProp.maxGridSize[2] << ")" << endl;
    cout << "Threads in warp          : " << ptrProp.warpSize << endl;
    cout << "===========================================" << endl;
    cout << endl;
    }

void printAllDeviceProp(void)
    {
    int nbDevice = nbDeviceDetect();
    for (int id = 0; id < nbDevice; id++)
	{
	cudaDeviceProp prop = deviceProp(id);
	print(prop, id);
	}
    }

/**
 * idDevice in [0,nbDevice-1]
 * Si le block est 2D, il s'agit du nombre de thread max pour tout le block 2D!
 */
int getMaxThreadPerBlock(int idDevice)
    {
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, idDevice));

    return prop.maxThreadsPerBlock;
    }

/**
 * of device actif
 */
int getMaxThreadPerBlock(void)
    {
    return getMaxThreadPerBlock(getIdCurrentDevive());
    }

/**
 * idDevice in [0,nbDevice-1]
 */
dim3 getMaxGridDim(int idDevice)
    {
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, idDevice));

    return dim3(prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }

/**
 * of device actif
 */
dim3 getMaxGridDim(void)
    {
    return getMaxGridDim(getIdCurrentDevive());
    }

/**
 * idDevice in [0,nbDevice-1]
 */
dim3 getMaxBlockDim(int idDevice)
    {
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, idDevice));

    return dim3(prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    }

/**
 * of device actif
 */
dim3 getMaxBlockDim(void)
    {
    return getMaxBlockDim(getIdCurrentDevive());
    }

int getIdCurrentDevive(void)
    {
    int deviceId;
    HANDLE_ERROR(cudaGetDevice(&deviceId));

    return deviceId;
    }

int getMultiprocessorCount(int idDevice)
    {
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, idDevice));

    return prop.multiProcessorCount;
    }

int getMultiprocessorCount(void)
    {
    return getMultiprocessorCount(getIdCurrentDevive());
    }

/**
 * Example: sm=1.2 major=1 minor=2
 */
int getCapacityMajor(int idDevice)
    {
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, idDevice));

    return prop.major;
    }

/**
 * Example: sm=1.2 major=1 minor=2
 */
int getCapacityMinor(int idDevice)
    {
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, idDevice));

    return prop.minor;
    }

/**
 * Example: sm=1.2 major=1 minor=2
 */
int getCapacityMajor(void)
    {
    return getCapacityMajor(getIdCurrentDevive());

    }

/**
 * Example: sm=1.2 major=1 minor=2
 */
int getCapacityMinor(void)
    {
    return getCapacityMinor(getIdCurrentDevive());
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

void cout01(int isTrue)
    {
    if (isTrue)
	cout << "True" << endl;
    else
	cout << "False" << endl;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

