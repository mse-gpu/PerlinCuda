#include <stdlib.h>
#include <iostream>

#include "GLUTWindowManagers.h"
#include "cuda_gl_interop.h"

#include "PerlinImage.hpp"

#include "cudaTools.h"
#include "deviceTools.h"

int bench(int argc, char** argv);
int launchApplication(int argc, char** argv);

int main(int argc, char** argv){
    //return launchApplication(argc, argv);
    return bench(argc, argv);
}

int launchApplication(int argc, char** argv){
    if (nbDeviceDetect() >= 1){
	int deviceId = 2;

	HANDLE_ERROR(cudaSetDevice(deviceId)); // active gpu of deviceId
	HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost)); // Not all gpu allow the use of mapMemory (avant prremier appel au kernel)
	HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

	GLUTWindowManagers::init(argc, argv);
	GLUTWindowManagers* glutWindowManager = GLUTWindowManagers::getInstance();

	GLImageFonctionelCudaSelections* image;

	std::cout << "Launch Perlin in Cuda" << std::endl;

	int w = 800;
	int h = 800;

	DomaineMaths domain(0, 0, w, h);

	image = new GLPerlinImage(w, h, domain);

	glutWindowManager->createWindow(image);
	glutWindowManager->runALL(); //Blocking

	delete image;

	return EXIT_SUCCESS;
    } else {
	return EXIT_FAILURE;
    }
}

#define DIM_H 1000
#define DIM_W 1000
#define TIMES 25

void launchPerlinAnimation(uchar4* ptrDevPixels, int w, int h, const DomaineMaths& domainNew, int t);

int bench(int argc, char** argv){
    std::cout << "Launch benchmark" << std::endl;

    if (nbDeviceDetect() >= 1){
	    int deviceId = 1;

	    HANDLE_ERROR(cudaSetDevice(deviceId)); // active gpu of deviceId
	    HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost)); // Not all gpu allow the use of mapMemory (avant prremier appel au kernel)

	    //Force the driver to run
	    uchar4* image;
	    HANDLE_ERROR(cudaMalloc(&image, DIM_W * DIM_H * sizeof(uchar4)));

	    std::cout << "End of malloc" << std::endl;
	    std::cout << "Size of the image: " << (DIM_W * DIM_H * sizeof(uchar4)) << std::endl;

	    CUevent start;
	    CUevent stop;
	    HANDLE_ERROR(cudaEventCreate(&start, CU_EVENT_DEFAULT));
	    HANDLE_ERROR(cudaEventCreate(&stop, CU_EVENT_DEFAULT));
	    HANDLE_ERROR(cudaEventRecord(start,0));

	    DomaineMaths domain(0, 0, DIM_W, DIM_H);

	    for(int i = 0; i < TIMES; ++i){
		launchPerlinAnimation(image, DIM_W, DIM_H, domain, 1);
	    }

	    float elapsed = 0;
	    HANDLE_ERROR(cudaEventRecord(stop,0));
	    HANDLE_ERROR(cudaEventSynchronize(stop));
	    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

	    std::cout << "Total (" << TIMES << " times) = " << elapsed << "ms" << std::endl;
	    std::cout << "Mean  (" << TIMES << " times) = " << (elapsed / TIMES) << "ms" << std::endl;

	    HANDLE_ERROR(cudaEventDestroy(start));
	    HANDLE_ERROR(cudaEventDestroy(stop));

	    HANDLE_ERROR(cudaFree(image));

	    return EXIT_SUCCESS;
    } else {
	    return EXIT_FAILURE;
    }
}
