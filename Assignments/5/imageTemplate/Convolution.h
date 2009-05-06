#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

extern "C"
{
    bool Wrapper_Convolution_Init(unsigned char * pRGBA, int w, int h, unsigned int pbo);
    bool Wrapper_Convolution_Release();
    bool Wrapper_Convolution_Run(int radius);
}

#endif