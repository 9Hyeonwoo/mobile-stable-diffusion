//
// Created by 구현우 on 2024/06/20.
//

#ifndef MY_OPENCL_GROUPNORMKERNEL_H
#define MY_OPENCL_GROUPNORMKERNEL_H

#include <CL/opencl.h>
#include <android/asset_manager.h>

class GroupNormKernel {
public:
    GroupNormKernel(cl_context context, cl_device_id deviceId, AAssetManager *assetManager);

    ~GroupNormKernel();

    cl_kernel local_reduction_mean;
    cl_kernel local_reduction_variance;
    cl_kernel group_norm;
};


#endif //MY_OPENCL_GROUPNORMKERNEL_H