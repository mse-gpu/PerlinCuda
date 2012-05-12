#include "NewtonImage.hpp"

extern void launchNewtonAnimation(uchar4* ptrDevPixels, int w, int h, const DomaineMaths& domainNew);

GLNewtonImage::GLNewtonImage(int dx, int dy, DomaineMaths domain): N(0), GLImageFonctionelCudaSelections(dx, dy, domain){
    //Nothing to init other than the initialization list
}

GLNewtonImage::~GLNewtonImage(){
    //Nothing
}

void GLNewtonImage::performKernel(uchar4* ptrDevPixels, int w, int h, const DomaineMaths& domainNew){
    launchNewtonAnimation(ptrDevPixels, w, h, domainNew);
}

void GLNewtonImage::idleFunc(){
    ++N;
    updateView();
}
