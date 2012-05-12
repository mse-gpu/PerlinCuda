#ifndef DEVICE_TOOLS_H_
#define DEVICE_TOOLS_H_

#include "cuda_runtime.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

int nbDeviceDetect(void);

int getIdCurrentDevive(void);

/**
 * idDevice in [0,nbDevice-1]
 */
cudaDeviceProp deviceProp(int idDevice);

/**
 * of device actif
 */
cudaDeviceProp deviceProp(void);

/**
 * idDevice in [0,nbDevice-1]
 */
void print(const cudaDeviceProp& ptrProp, int idDevice);
void printAllDeviceProp(void);

/**
 * idDevice in [0,nbDevice-1]
 * Si le block est 2D, il s'agit du nombre de thread max pour tout le block 2D!
 */
int getMaxThreadPerBlock(int idDevice);

/**
 * of device actif
 */
int getMaxThreadPerBlock(void);

/**
 * idDevice in [0,nbDevice-1]
 * dim3.x
 * dim3.y
 * dim3.z
 */
dim3 getMaxGridDim(int idDevice);

/**
 * idDevice in [0,nbDevice-1]
 * dim3.x
 * dim3.y
 * dim3.z
 */
dim3 getMaxBlockDim(int idDevice);

/**
 * of device actif
 */
dim3 getMaxBlockDim(void);

/**
 * of device actif
 */
dim3 getMaxGridDim(void);

int getMultiprocessorCount(void);
int getMultiprocessorCount(int idDevice);

/**
 * Example: sm=1.2 major=1 minor=2
 */
int getCapacityMajor(int idDevice);

/**
 * Example: sm=1.2 major=1 minor=2
 */
int getCapacityMinor(int idDevice);

/**
 * Example: sm=1.2 major=1 minor=2
 */
int getCapacityMajor(void);

/**
 * Example: sm=1.2 major=1 minor=2
 */
int getCapacityMinor(void);

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

