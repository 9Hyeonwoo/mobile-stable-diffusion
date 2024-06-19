//
// Created by 구현우 on 2024/06/20.
//

#ifndef MY_OPENCL_GEGLUKERNEL_H
#define MY_OPENCL_GEGLUKERNEL_H

#include <CL/opencl.h>
#include <android/asset_manager.h>

class GEGLUKernel {
public:
    GEGLUKernel(cl_context context, cl_device_id deviceId, AAssetManager *assetManager);

    ~GEGLUKernel();

    cl_kernel gelu_multiply;
};


#endif //MY_OPENCL_GEGLUKERNEL_H