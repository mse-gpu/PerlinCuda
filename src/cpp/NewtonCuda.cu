#include "Tools.hpp"

__global__ static void newtonAnimation(uchar4* ptrDevPixels, int w, int h, DomaineMaths domainNew);

__device__ static int newton(float x, float y);

void launchNewtonAnimation(uchar4* ptrDevPixels, int w, int h, const DomaineMaths& domainNew){
    dim3 blockPerGrid = dim3(32, 16, 1);
    dim3 threadPerBlock = dim3(32, 16, 1);

    newtonAnimation<<<blockPerGrid,threadPerBlock>>>(ptrDevPixels, w, h, domainNew);
}

__global__ static void newtonAnimation(uchar4* ptrDevPixels, int w, int h, DomaineMaths domainNew){
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    int nbThreadY = gridDim.y * blockDim.y;
    int nbThreadX = gridDim.x * blockDim.x;
    int nbThreadCuda = nbThreadY * nbThreadX;

    float dx = (float) (domainNew.dx / (float) w);
    float dy = (float) (domainNew.dy / (float) h);

    int tid = j +  (i * nbThreadX);

    float x, y;

    while(tid < (w * h)){
	int pixelI = tid / w;
	int pixelJ = tid - w * pixelI;

	x = domainNew.x0 + pixelJ * dx;
	y = domainNew.y0 + pixelI * dy;

	int color = newton(x, y);
	if(color == 0){
	    ptrDevPixels[tid].x = 0;
	    ptrDevPixels[tid].y = 0;
	    ptrDevPixels[tid].z = 0;
	} else if(color == 1){
	    ptrDevPixels[tid].x = 255;
	    ptrDevPixels[tid].y = 0;
	    ptrDevPixels[tid].z = 0;
	} else if(color == 2){
	    ptrDevPixels[tid].x = 0;
	    ptrDevPixels[tid].y = 255;
	    ptrDevPixels[tid].z = 0;
	} else if(color == 3){
	    ptrDevPixels[tid].x = 0;
	    ptrDevPixels[tid].y = 0;
	    ptrDevPixels[tid].z = 255;
	}

	ptrDevPixels[tid].w = 255;

	tid += nbThreadCuda;
    }
}

struct vector{
	float x;
	float y;
};

#define LIMIT 1000
#define PRECISION 1
#define CIRCLE 3
#define SQRT3 1.7320508075688772935

__device__ static bool near(float src, float target){
    float delta = src - target;

    if(delta < 0){
	delta = -delta;
    }

    if(delta <= PRECISION){
	return true;
    }

    return false;
}

__device__ static int newton(float x, float y){
    vector xn = {x,y};

    int current = 0;

    int times = 0;
    int last = 0;

    while(current < LIMIT){
    float fnx = xn.x * xn.x * xn.x - 3 * xn.x * xn.y * xn.y - 1;
    float fny = xn.y * xn.y * xn.y - 3 * xn.x * xn.x * xn.y;

    float ja = 3 * xn.x * xn.x - 3 * xn.y * xn.y;
    float jd = 3 * xn.y * xn.y - 3 * xn.x * xn.x;
    float jbc = 6 * xn.x * xn.y;

    float det = ja * jd - jbc * jbc; //det(A) = a*d - b*c

    float dx = (jd / det) * fnx + (jbc / det) * fny;
    float dy = (jbc / det) * fnx + (ja / det) * fny;

    xn.x = xn.x - dx;
    xn.y = xn.y - dy;

    if(near(xn.x, 1) && near(xn.y, 0)){
	if(times == CIRCLE && last == 1){
	    return 1;
	}

	if(last == 1){
	    ++times;
	} else {
	    times = 1;
	}

	last = 1;
    } else if(near(xn.x, -1/2) && near(xn.y, SQRT3 / 2)){
	if(times == CIRCLE && last == 2){
	    return 2;
	}

	if(last == 2){
	    ++times;
	} else {
	    times = 1;
	}

	last = 2;
    } else if(near(xn.x, -1/2) && near(xn.y, -SQRT3 / 2)){
	if(times == CIRCLE && last == 3){
	    return 3;
	}

	if(last == 3){
	    ++times;
	} else {
	    times = 1;
	}

	last = 3;
    } else {
	times = 0;
	last = 0;
    }

    ++current;
    }

    //Once we are here, it means that we are out the loop: black point
    return 0;
}
