//
// Created by 구현우 on 2024/04/20.
//

#ifndef MY_OPENCL_LAYERNORMKERNEL_H
#define MY_OPENCL_LAYERNORMKERNEL_H

#include <CL/opencl.h>
#include <android/asset_manager.h>

class LayerNormKernel {
public:
    LayerNormKernel(cl_context context, cl_device_id deviceId, AAssetManager *assetManager);

    ~LayerNormKernel();

    cl_kernel mean;
    cl_kernel variance;
    cl_kernel normalization;
};


#endif //MY_OPENCL_LAYERNORMKERNEL_H
