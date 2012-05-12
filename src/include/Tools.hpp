#ifndef GL_CUDA_TOOLS
#define GL_CUDA_TOOLS

#include "GLImageCudas.h"
#include "DomaineMaths.h"
#include "CalibreurCudas.h"

//Note: Moved to change it to static
__device__ static void HSB_TO_RVB(float H, float S, float V, unsigned char& R, unsigned char& G, unsigned char& B){
    float var_h = H * 6;
    unsigned char var_i = (unsigned char) var_h;

    float var_1 = V * (1 - S);
    float var_2 = V * (1 - S * (var_h - var_i));
    float var_3 = V * (1 - S * (1 - (var_h - var_i)));

    float var_r, var_g, var_b;
    if (var_i == 0)
	{
	var_r = V;
	var_g = var_3;
	var_b = var_1;
	}
    else if (var_i == 1)
	{
	var_r = var_2;
	var_g = V;
	var_b = var_1;
	}
    else if (var_i == 2)
	{
	var_r = var_1;
	var_g = V;
	var_b = var_3;
	}
    else if (var_i == 3)
	{
	var_r = var_1;
	var_g = var_2;
	var_b = V;
	}
    else if (var_i == 4)
	{
	var_r = var_3;
	var_g = var_1;
	var_b = V;
	}
    else
	{
	var_r = V;
	var_g = var_1;
	var_b = var_2;
	}

    //RGB results from 0 to 255
    R = (unsigned char) (var_r * 255);
    G = (unsigned char) (var_g * 255);
    B = (unsigned char) (var_b * 255);
}

#endif
