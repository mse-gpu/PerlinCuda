#ifndef GL_NEWTON_IMAGE
#define GL_NEWTON_IMAGE

#include <iostream>
#include "cudaTools.h"

#include "DomaineMaths.h"
#include "GLImageFonctionelCudaSelections.h"

class GLPerlinImage : public GLImageFonctionelCudaSelections {
    public:
	GLPerlinImage(int dx, int dy, DomaineMaths domain);
	virtual ~GLPerlinImage();

    protected:
	virtual void performKernel(uchar4* ptrDevPixels, int w, int h, const DomaineMaths& domainNew);
	virtual void idleFunc();

    private:
	int N;
};

#endif
