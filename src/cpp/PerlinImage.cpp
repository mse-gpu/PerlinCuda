#include "PerlinImage.hpp"

extern void launchPerlinAnimation(uchar4* ptrDevPixels, int w, int h, const DomaineMaths& domainNew);

GLPerlinImage::GLPerlinImage(int dx, int dy, DomaineMaths domain): N(0), GLImageFonctionelCudaSelections(dx, dy, domain){
    //Nothing to init other than the initialization list
}

GLPerlinImage::~GLPerlinImage(){
    //Nothing
}

void GLPerlinImage::performKernel(uchar4* ptrDevPixels, int w, int h, const DomaineMaths& domainNew){
    launchPerlinAnimation(ptrDevPixels, w, h, domainNew);
}

void GLPerlinImage::idleFunc(){
    ++N;
    updateView();
}
