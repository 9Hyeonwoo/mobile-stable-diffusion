//
// Created by 구현우 on 2024/04/20.
//

#ifndef MY_OPENCL_UTILKERNEL_H
#define MY_OPENCL_UTILKERNEL_H

#include <CL/opencl.h>
#include <android/asset_manager.h>

class UtilKernel {
public:
    UtilKernel(cl_context context, cl_device_id deviceId, AAssetManager *assetManager);

    ~UtilKernel();

    cl_kernel elemwise_add;
    cl_kernel permute3D_1_0_2;
    cl_kernel permute3D_0_2_1;
    cl_kernel gelu;
    cl_kernel softmax;
    cl_kernel silu;
};


#endif //MY_OPENCL_UTILKERNEL_H
