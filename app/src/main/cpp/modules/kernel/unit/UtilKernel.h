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
    cl_kernel gelu;
};


#endif //MY_OPENCL_UTILKERNEL_H
