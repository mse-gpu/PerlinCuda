#include "Tools.hpp"

__global__ static void perlinAnimation(uchar4* ptrDevPixels, int w, int h, DomaineMaths domainNew, int r1, int r2, int r3, int t);

void launchPerlinAnimation(uchar4* ptrDevPixels, int w, int h, const DomaineMaths& domainNew, int t){
    dim3 blockPerGrid = dim3(32, 16, 1);
    dim3 threadPerBlock = dim3(32, 16, 1);

    static int r1 = (rand() % 9000) + 1000;
    static int r2 = (rand() % 900000) + 100000;
    static int r3 = (rand() % 1000000000) + 1000000000;

    perlinAnimation<<<blockPerGrid,threadPerBlock>>>(ptrDevPixels, w, h, domainNew, r1, r2, r3, t);
}

__device__ static float perlinNoise(float x, float y, int r1, int r2, int r3);

__global__ static void perlinAnimation(uchar4* ptrDevPixels, int w, int h, DomaineMaths domainNew, int r1, int r2, int r3, int t){
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


	float c = perlinNoise(x+t,y, r1, r2, r3);

	ptrDevPixels[tid].x = 135;
	ptrDevPixels[tid].y = 206;
	ptrDevPixels[tid].z = 250;
	ptrDevPixels[tid].w = c * 255.0;

	tid += nbThreadCuda;
    }
}

__device__ static float interpolate(float x, float y, float a){
    float val = (1 - cos(a * M_PI)) * 0.5;
    return x * (1 - val) + y * val;
}

__device__ static float noise(int x, int y, int r1, int r2, int r3){
    int n = x + y * 57;
    n = (n << 13) ^ n;

    return (1.0 - ((n * (n * n * r1 + r2) + r3) & 0x7fffffff) / 1073741824.0);
}

__device__ static float smooth(float x, float y, int r1, int r2, int r3){
    float n1 = noise((int)x, (int)y, r1, r2, r3);
    float n2 = noise((int)x + 1, (int)y, r1, r2, r3);
    float n3 = noise((int)x, (int)y + 1, r1, r2, r3);
    float n4 = noise((int)x + 1, (int)y + 1, r1, r2, r3);

    float i1 = interpolate(n1, n2, x - (int)x);
    float i2 = interpolate(n3, n4, x - (int)x);

    return interpolate(i1, i2, y - (int)y);
}

__device__ static float perlinNoise(float x, float y, int r1, int r2, int r3){
    float total = 0.0;

    float frequency = 0.015;
    float persistence = 0.65;
    float octaves = 16;
    float amplitude = 1;

    for(int lcv = 0; lcv < octaves; ++lcv){
	total += smooth(x * frequency, y * frequency, r1, r2, r3) * amplitude;
	frequency *= 2;
	amplitude *= persistence;
    }

    const float cloudCoverage = 0;
    const float cloudDensity = 1;

    total = (total + cloudCoverage) * cloudDensity;

    if(total <  0){
	return 0.0;
    } else if(total > 1.0){
	return 1.0;
    } else {
	return total;
    }
}
