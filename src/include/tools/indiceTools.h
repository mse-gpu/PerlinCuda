#ifndef INDICETOOLS_H_
#define INDICETOOLS_H_


/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * j=j(threadIdx,blockDim,blockIdx)
 * j in [0,m-1]
 * j=threadIdx.x + (blockDim.x * blockIdx.x)
 */
__device__ static int j0()
    {
    return threadIdx.x + (blockDim.x * blockIdx.x);
    }

/**
 * i=i(threadIdx,blockDim,blockIdx)
 * i in [0,n-1]
 * i=threadIdx.y + (blockDim.y * blockIdx.y)
 */
__device__ static int i0()
    {
    return threadIdx.y + (blockDim.y * blockIdx.y);
    }

/**
 * i in [0,n-1]
 * j in [0,m-1]
 * s in [0,nm-1]
 * s=i*m+j
 */
__device__ static int s0(int i,int j,int m)
    {
    return i*m+j;
    }

/**
 * tid in [0,n-1]
 * tid=threadIdx.x + (blockDim.x * blockIdx.x)
 */
__device__ static int tid0()
    {
    return threadIdx.x + (blockDim.x * blockIdx.x);
    }

/**
 * nbThreadTotalCudaX=gridDim.x * blockDim.x
 */
__device__  static int nbThreadTotalCuda()
    {
    return (gridDim.x *gridDim.y) * (blockDim.x*blockDim.y*blockDim.z); // dim=1 si pas use
    }

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
